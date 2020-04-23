using System;
using System.Collections.Concurrent;
using System.Data;
using System.Diagnostics;
using System.Threading.Tasks;
using Dynamic.MonteCarloSimulation.Properties;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;

namespace Dynamic.MonteCarloSimulation.CrackPOF_MD
{
	/// <summary>
	/// Implements Crack POF, MD calculator
	/// </summary>
	/// <seealso cref="Dynamic.MonteCarloSimulation.SimulationCalculator{Dynamic.MonteCarloSimulation.CrackPOF_MD.ICrackPOFMDCalculatorInputs, Dynamic.MonteCarloSimulation.CrackPOF_MD.ICrackPOFMDCalculatorOutputs, Dynamic.MonteCarloSimulation.ISimulationIntermediates}" />
	public class CrackPOFMDCalculator
        : SimulationCalculator<ICrackPOFMDCalculatorInputs, ICrackPOFMDCalculatorOutputs, ISimulationIntermediates>
    {
        #region Constants
        // wall thickness, mean
        private const double c_Wt_m = 1.01;
        // wall thickness, sd
        private const double c_Wt_sd = 0.01;
        // diameter, mean
        private const double c_D_m = 1;
        // diameter, sd
        private const double c_D_sd = 0.0006;
        // grade, mean
        private const double c_S_m = 1.1;
        // grade, sd
        private const double c_S_sd = 0.035;
        // Young's Modulus (psi)
        private const double c_Young = 30000000;
        // Young's Modulus, mean
        private const double c_Young_m = 1;
        // Young's Modulus, sd
        private const double c_Young_sd = 0.04;
        // Fracture Area (in2)
        private const double c_FractureArea = 0.124;
        // MD Crack Depth Mean (fraction)
        private const double c_CrackDepthMean = 1;
        // MD Crack Depth, sd, EMAT, WT<10mm (fraction)
        private const double c_MDCD_sd_EMAT_wt_LT10 = 0.117;
        // MD Crack Depth, sd, EMAT, WT>=10mm (fraction)
        private const double c_MDCD_sd_EMAT_wt_GT10 = 0.156;
        // MD Crack Depth, sd, AFD (fraction)
        private const double c_MDCD_sd_AFD = 0.195;
        // MD Crack Length Mean (fraction)
        private const double c_CrackLengthMean = 1;
        // MD Crack Length SD, EMAT (mm)
        private const double c_MDCL_sd_EMAT = 7.8;
        // MD Crack Length SD, AFD (mm)
        private const double c_MDCL_sd_AFD = 15.6;

        // indexes for flags arrays
        private const int FlagsArrayCount = 4;
        private const int LeakFlags_index = 0;
        private const int RuptureFlags_index = 1;
        private const int LeakOrRuptureFlags_index = 2;
        private const int Iterations_index = 3;
        #endregion

        #region Methods
        /// <summary>
        /// Determines if algorithm can be calculated based on passed inputs..
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <returns>
        ///   <c>true</c> if algorithm can be calculated; otherwise, <c>false</c>.
        /// </returns>
        private bool CanCalculate(ICrackPOFMDCalculatorInputs inputs)
        {
            // certain values must be present
            return (inputs.InstallationDate != null) &&
                (inputs.ILIDate != null) &&
                (inputs.NominalWallThickness_mm != null) &&
                (inputs.OutsideDiameter_in != null) &&
                (inputs.PipeGrade_MPa != null) &&
                (inputs.MaximumAllowableOperatingPressure_kPa != null);
        }

		/// <summary>
		/// Calculates the single row.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="intermediates">The target to store intermediates in simulation. Optional, if not passed, then no intermediates will be stored.</param>
		/// <returns>
		/// Class with output results.
		/// </returns>
		/// <exception cref="System.ArgumentNullException">Thrown when <paramref name="inputs" /> parameter is <b>null</b>.</exception>
		protected override ICrackPOFMDCalculatorOutputs CalculateSingleRow(
			ICrackPOFMDCalculatorInputs inputs,
			ConcurrentBag<ISimulationIntermediates> intermediates = null
			)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            ICrackPOFMDCalculatorOutputs outputs = new CrackPOFMDCalculatorOutputs();

            // check if we can calculate it
            if (!CanCalculate(inputs))
            {
                // mark as not calculated
                outputs.Calculated = false;
                outputs.ErrorMessage = MCS_MISSED_INPUTS;
                return outputs;
            }

            // start timer
            Stopwatch timer = new Stopwatch();
            timer.Start();

            // get inputs
            int numberOfIterations = inputs.NumberOfIterations;
            int numberOfYears = inputs.NumberOfYears;

            double wt = inputs.NominalWallThickness_mm != null ? (double)inputs.NominalWallThickness_mm : 0;
            double od = inputs.OutsideDiameter_in != null ? (double)inputs.OutsideDiameter_in : 0;
            double ILIDefectL = inputs.ILIMeasuredClusterDefectLength_mm != null ? (double)inputs.ILIMeasuredClusterDefectLength_mm : 0;
            double PeakDepth = inputs.ILIMeasuredClusterDefectDepthPrc != null ? (double)inputs.ILIMeasuredClusterDefectDepthPrc : 0;
            double pipeGrade = inputs.PipeGrade_MPa != null ? (double)inputs.PipeGrade_MPa : 0;
            double maop = inputs.MaximumAllowableOperatingPressure_kPa != null ? (double)inputs.MaximumAllowableOperatingPressure_kPa : 0;
            DateTime pipeInstallDate = inputs.InstallationDate.Value;
            DateTime iliDate = inputs.ILIDate.Value;

            string iliCompany = inputs.ILICompany;

            double? pipeToughness = inputs.SeamPipeToughness;

            // Toughness, imperial
            double toughnessX;
            if (pipeToughness != null)
            {
                toughnessX = (double)pipeToughness * Convert_Joule_To_FootPound;
            }
            else
            {
                toughnessX = ToughnessByYearDiameter(pipeInstallDate.Year, od) * Convert_Joule_To_FootPound;
            }

            // Grade, imperial (psi)
            double Si = pipeGrade * Convert_MPa_PSI;

            // Wall thickness, imperial (in)
            double Wti = wt * Convert_MM_to_Inch;

            // MAOP, imperial (psi)
            double MAOPi = maop * Convert_kPa_PSI;

            // MD Crack Length, SD (mm)
            double MDCL_sd = 8.034;

            // MD Crack Depth, SD (mm)
            double MDCD_sd = wt * 0.121;

            // MD Crack Depth Measured (in)
            double MDCrackDepthM = PeakDepth * wt * Convert_MM_to_Inch;

            // create instance of failure flags...
            ConcurrentBag<double[,]> allFlags = new ConcurrentBag<double[,]>();

            //  This Parallel.For() construct accumulates counts in the TFailureFlags class and periodically (but not every
            //  iteration) updates the flags instance from the threadflags instance and creates a new threadflags instance.
            Parallel.For(0, numberOfIterations,
                () => new double[FlagsArrayCount, 1],
                (j, loop, threadflags) =>
                {
                    // Young's Modulus, distributed (psi)
                    double Young_p = Normal.InvCDF(
                        c_Young * c_Young_m,
                        c_Young * c_Young_sd,
						SystemRandomSource.Default.NextDouble());

                    // Grade, distributed (in)
                    double S_p = Normal.InvCDF(
                        Si * c_S_m,
                        Si * c_S_sd,
						SystemRandomSource.Default.NextDouble());

                    // Outside Diameter, distributed (in)
                    double D_p = Normal.InvCDF(
                        od * c_D_m,
                        od * c_D_sd,
						SystemRandomSource.Default.NextDouble());

                    // Wall thickness, distributed (in)
                    double Wt_p = Normal.InvCDF(
                        Wti * c_Wt_m,
                        Wti * c_Wt_sd,
						SystemRandomSource.Default.NextDouble());

                    // MD Crack Length, Run (in)
                    double MDCrackLengthRun = Math.Max(0,
                        Normal.InvCDF(
                            ILIDefectL * c_CrackLengthMean,
                            MDCL_sd,
							SystemRandomSource.Default.NextDouble())) * Convert_MM_to_Inch;

                    // MDCrack Length Growth Rate (in/yr)
                    double MDCrackLengthGR = MDCrackLengthRun / ((iliDate - pipeInstallDate).TotalDays / 365.25);

                    // MD Crack Length (in)
                    double MDCrackLength = MDCrackLengthRun;

                    // MD Crack Depth, Run (in)
                    double MDCrackDepthRun = Math.Max(0,
                        Normal.InvCDF(
                            MDCrackDepthM * c_CrackDepthMean,
                            MDCD_sd * Convert_MM_to_Inch,
							SystemRandomSource.Default.NextDouble()));

                    // MD Crack Depth Growth Rate (in/yr)
                    double MDCrackDepthGR = MDCrackDepthRun / ((iliDate - pipeInstallDate).TotalDays / 365.25);

                    // MD Crack Depth (in)
                    double MDCrackDepth = MDCrackDepthRun + (DateTime.Now - iliDate).TotalDays / 365.25 * MDCrackDepthGR;

                    // Elliptical C Equivalent Shape
                    double MDEllipticalC = Math.Min(Math.PI * MDCrackLength / 4, Math.Sqrt(67.2 * D_p * Wt_p / 2));

                    // CRT Ratio
                    double MDCRTRatio = Math.Min(93.753, Math.Pow(MDEllipticalC, 2) / (D_p * Wt_p / 2));

                    // MD Flow Stress (psi)
                    double MDFlowStress = S_p + 10000;

                    // MD Y
                    double MDY = ((12 * Young_p * Math.PI * toughnessX / c_FractureArea) / (8 * MDEllipticalC * Math.Pow(MDFlowStress, 2))) * Math.Pow(1 - Math.Pow(MDCrackDepth / Wt_p, 0.8), -1);

                    // MD X
                    double MDX = (12 * Young_p * Math.PI * toughnessX / c_FractureArea) / (8 * MDEllipticalC * Math.Pow(MDFlowStress, 2));

                    // MD Mt
                    double MDMt = Math.Max(MDCrackDepth / Wt_p + 0.001, Math.Sqrt(1 + 1.255 * MDCRTRatio - 0.0135 * Math.Pow(MDCRTRatio, 2)));

                    // MD MP
                    double MDMP = (1 - MDCrackDepth / (Wt_p * MDMt)) / (1 - MDCrackDepth / Wt_p);

                    // MD Failure Stress (psi)
                    double MDFailureStress = (MDFlowStress / MDMP) * Math.Acos(Math.Exp(-MDX)) / Math.Acos(Math.Exp(-MDY));

                    // Predicted Failure Pressure (psi)
                    double PredictFailPressure = 2 * MDFailureStress * Wt_p / D_p;

                    // Rupture flag
                    int ruptureFlag = PredictFailPressure <= MAOPi ? 1 : 0;

                    // Leak flag
                    int leakFlag = MDCrackDepth >= Wt_p ? 1 : 0;

                    // Increment cumulative flags
                    threadflags[LeakFlags_index, 0] += leakFlag;
                    threadflags[RuptureFlags_index, 0] += ruptureFlag;
                    threadflags[LeakOrRuptureFlags_index, 0] += Math.Max(ruptureFlag, leakFlag);
                    threadflags[Iterations_index, 0] += 1;  //  Sanity check that we're not losing results due to Parallel.

                    // return calculated results
                    return threadflags;
                },
                (threadflags) =>
                {
                    allFlags.Add(threadflags);
                });

            // now combine all flags in one object
            double[,] flags = new double[FlagsArrayCount, 1];
            while (!allFlags.IsEmpty)
            {
                double[,] item;
                if (allFlags.TryTake(out item))
                {
                    flags[LeakFlags_index, 0] += item[LeakFlags_index, 0];
                    flags[RuptureFlags_index, 0] += item[RuptureFlags_index, 0];
                    flags[LeakOrRuptureFlags_index, 0] += item[LeakOrRuptureFlags_index, 0];
                    flags[Iterations_index, 0] += item[Iterations_index, 0];
                }
            }

            //  Convert flags to probabilities, and determine failures
            // don't forget to convert it in DOUBLE...
            double leakProbability = flags[LeakFlags_index, 0] * 1.0 / flags[Iterations_index, 0];
            outputs.POE_Leak = leakProbability;
            double ruptureProbability = flags[RuptureFlags_index, 0] * 1.0 / flags[Iterations_index, 0];
            outputs.POE_Rupture = ruptureProbability;
            double failureProbability = Math.Max(leakProbability, ruptureProbability);
            outputs.FailureProbability = failureProbability;

            // mark record as calculated
            outputs.Calculated = true;

            // set time stamp
            timer.Stop();
            outputs.Elapsed = timer.Elapsed.TotalMilliseconds;

            return outputs;
        }

        /// <summary>
        /// Loads the input parameters.
        /// </summary>
        /// <param name="row"></param>
        /// <returns>
        /// Inputs class with data from passed data row.
        /// </returns>
        /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="row"/> parameter is <b>null</b>.</exception>
        protected override ICrackPOFMDCalculatorInputs LoadInputs(DataRow row)
        {
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            ICrackPOFMDCalculatorInputs inputs = new CrackPOFMDCalculatorInputs();

            inputs.NumberOfIterations = (int)row[nameof(ICrackPOFMDCalculatorInputs.NumberOfIterations)];
            inputs.NumberOfYears = (int)row[nameof(ICrackPOFMDCalculatorInputs.NumberOfYears)];
            inputs.Id = Convert.ToInt32(row[nameof(ICrackPOFMDCalculatorInputs.Id)].ToString());
            inputs.RangeId = Convert.ToInt32(row[nameof(ICrackPOFMDCalculatorInputs.RangeId)].ToString());
            inputs.NominalWallThickness_mm = row[nameof(ICrackPOFMDCalculatorInputs.NominalWallThickness_mm)].IfNullable<double>();
            inputs.OutsideDiameter_in = row[nameof(ICrackPOFMDCalculatorInputs.OutsideDiameter_in)].IfNullable<double>();
            inputs.PipeGrade_MPa = row[nameof(ICrackPOFMDCalculatorInputs.PipeGrade_MPa)].IfNullable<double>();
            inputs.Toughness_Joule = row[nameof(ICrackPOFMDCalculatorInputs.Toughness_Joule)].IfNullable<double>();
            inputs.MaximumAllowableOperatingPressure_kPa = row[nameof(ICrackPOFMDCalculatorInputs.MaximumAllowableOperatingPressure_kPa)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectLength_mm = row[nameof(ICrackPOFMDCalculatorInputs.ILIMeasuredClusterDefectLength_mm)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectDepthPrc = row[nameof(ICrackPOFMDCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)].IfNullable<double>();
            inputs.SurfaceIndicator = row[nameof(ICrackPOFMDCalculatorInputs.SurfaceIndicator)].ToString();
            inputs.SeamPipeToughness = row[nameof(ICrackPOFMDCalculatorInputs.SeamPipeToughness)].IfNullable<double>();
            inputs.InstallationDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(ICrackPOFMDCalculatorInputs.InstallationDate)].ToString());
            inputs.ILIDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(ICrackPOFMDCalculatorInputs.ILIDate)].ToString());
            inputs.ILICompany = row[nameof(ICrackPOFMDCalculatorInputs.ILICompany)].ToString();

            return inputs;
        }

        /// <summary>
        /// Creates the output table columns.
        /// </summary>
        /// <param name="inputsTable">The inputs table.</param>
        /// <param name="outputsTable">The outputs table.</param>
        /// <param name="outputs">The outputs.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputsTable" />, <paramref name="outputsTable" />, <paramref name="outputs"/> parameter is <b>null</b>.</exception>
        protected override void CreateOutputTableColumns(DataTable inputsTable, DataTable outputsTable, ICrackPOFMDCalculatorOutputs outputs)
        {
            if (inputsTable == null)
            {
                throw new ArgumentNullException(nameof(inputsTable));
            }
            if (outputsTable == null)
            {
                throw new ArgumentNullException(nameof(outputsTable));
            }
            if (outputs == null)
            {
                throw new ArgumentNullException(nameof(outputs));
            }

            // add all columns from input
            for (int i = 0; i < inputsTable.Columns.Count; i++)
            {
                outputsTable.Columns.Add(inputsTable.Columns[i].ColumnName, inputsTable.Columns[i].DataType);
            }

            // create columns for common data
            outputsTable.Columns.Add(nameof(ICrackPOFMDCalculatorOutputs.Elapsed), typeof(double));
            outputsTable.Columns.Add(nameof(ICrackPOFMDCalculatorOutputs.Calculated), typeof(string));
            outputsTable.Columns.Add(nameof(ICrackPOFMDCalculatorOutputs.ErrorMessage), typeof(string));

            outputsTable.Columns.Add(nameof(ICrackPOFMDCalculatorOutputs.POE_Leak), typeof(double));
            outputsTable.Columns.Add(nameof(ICrackPOFMDCalculatorOutputs.POE_Rupture), typeof(double));
            outputsTable.Columns.Add(nameof(ICrackPOFMDCalculatorOutputs.FailureProbability), typeof(double));
        }

        /// <summary>
        /// Saves the input and outputs in passed data row.
        /// </summary>
        /// <param name="inputs">The inputs row.</param>
        /// <param name="outputs">The outputs.</param>
        /// <param name="outputsRow">The row.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputs"/>, <paramref name="outputs"/>, <paramref name="outputsRow"/> parameter is <b>null</b>.</exception>
        protected override void SaveInputOutputs(ICrackPOFMDCalculatorInputs inputs, ICrackPOFMDCalculatorOutputs outputs, DataRow outputsRow)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }
            if (outputs == null)
            {
                throw new ArgumentNullException(nameof(outputs));
            }
            if (outputsRow == null)
            {
                throw new ArgumentNullException(nameof(outputsRow));
            }

            // output inputs
            outputsRow[nameof(ICrackPOFMDCalculatorInputs.NumberOfIterations)] = inputs.NumberOfIterations;
            outputsRow[nameof(ICrackPOFMDCalculatorInputs.NumberOfYears)] = inputs.NumberOfYears;
            outputsRow[nameof(ICrackPOFMDCalculatorInputs.Id)] = inputs.Id;
            if (inputs.NominalWallThickness_mm != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.NominalWallThickness_mm)] = inputs.NominalWallThickness_mm;
            }
            if (inputs.OutsideDiameter_in != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.OutsideDiameter_in)] = inputs.OutsideDiameter_in;
            }
            if (inputs.PipeGrade_MPa != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.PipeGrade_MPa)] = inputs.PipeGrade_MPa;
            }
            if (inputs.Toughness_Joule != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.Toughness_Joule)] = inputs.Toughness_Joule;
            }
            if (inputs.MaximumAllowableOperatingPressure_kPa != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.MaximumAllowableOperatingPressure_kPa)] = inputs.MaximumAllowableOperatingPressure_kPa;
            }
            if (inputs.ILIMeasuredClusterDefectLength_mm != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.ILIMeasuredClusterDefectLength_mm)] = inputs.ILIMeasuredClusterDefectLength_mm;
            }
            if (inputs.ILIMeasuredClusterDefectDepthPrc != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)] = inputs.ILIMeasuredClusterDefectDepthPrc;
            }
            outputsRow[nameof(ICrackPOFMDCalculatorInputs.SurfaceIndicator)] = inputs.SurfaceIndicator;
            if (inputs.SeamPipeToughness != null)
            {
                outputsRow[nameof(ICrackPOFMDCalculatorInputs.SeamPipeToughness)] = inputs.SeamPipeToughness;
            }
            outputsRow[nameof(ICrackPOFMDCalculatorInputs.InstallationDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.InstallationDate);
            outputsRow[nameof(ICrackPOFMDCalculatorInputs.ILIDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.ILIDate);
            outputsRow[nameof(ICrackPOFMDCalculatorInputs.ILICompany)] = inputs.ILICompany;

            // output common data
            outputsRow[nameof(ICrackPOFMDCalculatorOutputs.Elapsed)] = outputs.Elapsed;
            outputsRow[nameof(ICrackPOFMDCalculatorOutputs.Calculated)] = outputs.Calculated ? "Y" : "N";
            outputsRow[nameof(ICrackPOFMDCalculatorOutputs.ErrorMessage)] = outputs.ErrorMessage;

            outputsRow[nameof(ICrackPOFMDCalculatorOutputs.POE_Leak)] = outputs.POE_Leak;
            outputsRow[nameof(ICrackPOFMDCalculatorOutputs.POE_Rupture)] = outputs.POE_Rupture;
            outputsRow[nameof(ICrackPOFMDCalculatorOutputs.FailureProbability)] = outputs.FailureProbability;
        }
        #endregion
    }
}
