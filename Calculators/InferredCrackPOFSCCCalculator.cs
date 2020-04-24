using System;
using System.Collections.Concurrent;
using System.Data;
using System.Diagnostics;
using System.Threading.Tasks;
using Dynamic.MonteCarloSimulation.Properties;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;

namespace Dynamic.MonteCarloSimulation.Inferred_CrackPOF_SCC
{
	/// <summary>
	/// Implements Inferred Crack POF, SCC calculator
	/// </summary>
	/// <seealso cref="Dynamic.MonteCarloSimulation.SimulationCalculator{Dynamic.MonteCarloSimulation.Inferred_CrackPOF_SCC.IInferredCrackPOFSCCCalculatorInputs, Dynamic.MonteCarloSimulation.Inferred_CrackPOF_SCC.IInferredCrackPOFSCCCalculatorOutputs, Dynamic.MonteCarloSimulation.ISimulationIntermediates}" />
	public class InferredCrackPOFSCCCalculator
        : SimulationCalculator<IInferredCrackPOFSCCCalculatorInputs, IInferredCrackPOFSCCCalculatorOutputs, ISimulationIntermediates>
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
        // SCC Crack Depth, mean (fraction)
        private const double c_CD_m = 1;
        // SCC Crack Depth, sd, Rosen, WT<10mm (fraction)
        private const double c_SCCCD_sd_Rosen_wt_LT10 = 0.117;
        // SCC Crack Depth, sd, Rosen, WT>=10mm (fraction)
        private const double c_SCCCD_sd_Rosen_wt_GT10 = 0.156;
        // SCC Crack Depth, sd, PII, (mm)
        private const double c_SCCCD_sd_PII = 0.31;
        // SCC Crack Length, mean (fraction)
        private const double c_CL_m = 1;
        // SCC Crack Length, sd, Rosen (mm)
        private const double c_SCCCL_sd_Rosen = 7.8;
        // SCC Crack Length, sd, PII (mm)
        private const double c_SCCCL_sd_PII = 6.1;
        // Inferred Crack Depth, sd, (mm)
        private const double c_CD_sd_default = 0.310;
        // Inferred Crack Length, sd, (mm)
        private const double c_CL_sd_default = 6.1;

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
        private bool CanCalculate(IInferredCrackPOFSCCCalculatorInputs inputs)
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
		/// Instance of <see cref="IInferredCrackPOFSCCCalculatorOutputs" /> with calculation results.
		/// </returns>
		/// <exception cref="System.ArgumentNullException">Thrown when <paramref name="inputs" /> parameter is <b>null</b>.</exception>
		protected override IInferredCrackPOFSCCCalculatorOutputs CalculateSingleRow(
			IInferredCrackPOFSCCCalculatorInputs inputs,
			ConcurrentBag<ISimulationIntermediates> intermediates = null
			)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            IInferredCrackPOFSCCCalculatorOutputs outputs = new InferredCrackPOFSCCCalculatorOutputs();

            // check if we can calculate it
            if (!CanCalculate(inputs))
            {
                // mark as not calculated
                outputs.Calculated = false;
                outputs.ErrorMessage = MCS_MISSED_INPUTS;
                return outputs;
            }

            // start timer
            var timer = new Stopwatch();
            timer.Start();

            // get inputs
            int numberOfIterations = inputs.NumberOfIterations;
            int numberOfYears = inputs.NumberOfYears;

            double wt = inputs.NominalWallThickness_mm != null ? (double)inputs.NominalWallThickness_mm : 0;
            double od = inputs.OutsideDiameter_in != null ? (double)inputs.OutsideDiameter_in : 0;
            double ILIDefectL = inputs.ILIMeasuredClusterDefectLength_mm != null ? (double)inputs.ILIMeasuredClusterDefectLength_mm : 0;
            double peakDepth = inputs.ILIMeasuredClusterDefectDepthPrc != null ? (double)inputs.ILIMeasuredClusterDefectDepthPrc : 0;
            double pipeGrade = inputs.PipeGrade_MPa != null ? (double)inputs.PipeGrade_MPa : 0;
            double? pipeToughness = inputs.Toughness_Joule;
            double maop = inputs.MaximumAllowableOperatingPressure_kPa != null ? (double)inputs.MaximumAllowableOperatingPressure_kPa : 0;
            DateTime pipeInstallDate = inputs.InstallationDate.Value;
            DateTime iliDate = inputs.ILIDate.Value;

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

            // Crack Length, SD (mm)
            double CL_sd = c_CL_sd_default;

            // Crack Depth, SD (mm)
            double CD_sd = c_CD_sd_default;

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

                    // Crack Length (in)
                    double CrackLength = Math.Max(1, Normal.InvCDF(ILIDefectL * c_CL_m, CL_sd, SystemRandomSource.Default.NextDouble())) * Convert_MM_to_Inch;

                    // Crack Depth 1 Growth Rate (in/yr)
                    double CrackDepthGR1 = 0.1 * Math.Pow(-Math.Log(1 - SystemRandomSource.Default.NextDouble()), 1 / 2.55) * Convert_MM_to_Inch;

                    // Crack Depth, Run (in)
                    double CrackDepthRun1 = Math.Max(0, Normal.InvCDF(peakDepth * wt * c_CD_m, CD_sd, SystemRandomSource.Default.NextDouble())) * Convert_MM_to_Inch;

                    // Crack Depth (in)
                    double CrackDepth1 = CrackDepthRun1 + (DateTime.Now - iliDate).TotalDays / 365.25 * CrackDepthGR1;

                    // Elliptical C Equivalent Shape
                    double EllipticalC = Math.Min(Math.PI * CrackLength / 4, Math.Sqrt(67.2 * D_p * Wt_p / 2));

                    // CRT Ratio
                    double CRTRatio1 = Math.Min(67.2, Math.Pow(EllipticalC, 2) / (D_p * Wt_p / 2));

                    // Flow Stress (psi)
                    double FlowStress1 = S_p + 10000;

                    // Y
                    double Y1 = ((12 * Young_p * Math.PI * toughnessX / c_FractureArea) / (8 * EllipticalC * Math.Pow(FlowStress1, 2))) * Math.Pow(1 - Math.Pow(CrackDepth1 / Wt_p, 0.8), -1);

                    // X
                    double X1 = (12 * Young_p * Math.PI * toughnessX / c_FractureArea) / (8 * EllipticalC * Math.Pow(FlowStress1, 2));

                    // Mt
                    double Mt1 = Math.Max(CrackDepth1 / Wt_p + 0.001, Math.Pow(1 + 1.255 * CRTRatio1 - 0.0135 * Math.Pow(CRTRatio1, 2), 0.5));

                    // MP
                    double MP1 = (1 - CrackDepth1 / (Wt_p * Mt1)) / (1 - CrackDepth1 / Wt_p);

                    // Failure Stress (psi)
                    double FailureStress1 = (FlowStress1 / MP1) * Math.Acos(Math.Exp(-X1)) / Math.Acos(Math.Exp(-Y1));

                    // Failure Pressure (psi)
                    double CrackFailPressure1 = 2 * FailureStress1 * Wt_p / D_p;

                    // Leak flag
                    // For a purpose of this algorithm - treat everything as leaks
                    int leakFlag = (CrackFailPressure1 <= MAOPi) || (CrackDepth1 >= (0.8 * wt * Convert_MM_to_Inch)) ? 1 : 0;

                    //  Increment cumulative flags
                    threadflags[LeakFlags_index, 0] += leakFlag;
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
                if (allFlags.TryTake(out double[,] item))
                {
                    flags[LeakFlags_index, 0] += item[LeakFlags_index, 0];
                    flags[RuptureFlags_index, 0] += item[RuptureFlags_index, 0];
                    flags[LeakOrRuptureFlags_index, 0] += item[LeakOrRuptureFlags_index, 0];
                    flags[Iterations_index, 0] += item[Iterations_index, 0];
                }
            }

            // Convert flags to probabilities, and determine failures
            // don't forget to convert it in DOUBLE...
            // For a purpose of this algorithm - treat everything as leaks
            double crackFailureProbability = flags[LeakFlags_index, 0] * 1.0 / flags[Iterations_index, 0];
            outputs.FailureProbability = crackFailureProbability;

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
        protected override IInferredCrackPOFSCCCalculatorInputs LoadInputs(DataRow row)
        {
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            IInferredCrackPOFSCCCalculatorInputs inputs = new InferredCrackPOFSCCCalculatorInputs();

            inputs.NumberOfIterations = (int)row[nameof(IInferredCrackPOFSCCCalculatorInputs.NumberOfIterations)];
            inputs.NumberOfYears = (int)row[nameof(IInferredCrackPOFSCCCalculatorInputs.NumberOfYears)];
            inputs.Id = Convert.ToInt32(row[nameof(IInferredCrackPOFSCCCalculatorInputs.Id)].ToString());
            inputs.RangeId = Convert.ToInt32(row[nameof(IInferredCrackPOFSCCCalculatorInputs.RangeId)].ToString());
            inputs.NominalWallThickness_mm = row[nameof(IInferredCrackPOFSCCCalculatorInputs.NominalWallThickness_mm)].IfNullable<double>();
            inputs.OutsideDiameter_in = row[nameof(IInferredCrackPOFSCCCalculatorInputs.OutsideDiameter_in)].IfNullable<double>();
            inputs.PipeGrade_MPa = row[nameof(IInferredCrackPOFSCCCalculatorInputs.PipeGrade_MPa)].IfNullable<double>();
            inputs.Toughness_Joule = row[nameof(IInferredCrackPOFSCCCalculatorInputs.Toughness_Joule)].IfNullable<double>();
            inputs.MaximumAllowableOperatingPressure_kPa = row[nameof(IInferredCrackPOFSCCCalculatorInputs.MaximumAllowableOperatingPressure_kPa)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectLength_mm = row[nameof(IInferredCrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectLength_mm)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectDepthPrc = row[nameof(IInferredCrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)].IfNullable<double>();
            inputs.SurfaceIndicator = row[nameof(IInferredCrackPOFSCCCalculatorInputs.SurfaceIndicator)].ToString();
            inputs.InstallationDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(IInferredCrackPOFSCCCalculatorInputs.InstallationDate)].ToString());
            inputs.ILIDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(IInferredCrackPOFSCCCalculatorInputs.ILIDate)].ToString());

            return inputs;
        }

        /// <summary>
        /// Creates the output table columns.
        /// </summary>
        /// <param name="inputsTable">The inputs table.</param>
        /// <param name="outputsTable">The outputs table.</param>
        /// <param name="outputs">The outputs.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputsTable" />, <paramref name="outputsTable" />, <paramref name="outputs"/> parameter is <b>null</b>.</exception>
        protected override void CreateOutputTableColumns(DataTable inputsTable, DataTable outputsTable, IInferredCrackPOFSCCCalculatorOutputs outputs)
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
            outputsTable.Columns.Add(nameof(IInferredCrackPOFSCCCalculatorOutputs.Elapsed), typeof(double));
            outputsTable.Columns.Add(nameof(IInferredCrackPOFSCCCalculatorOutputs.Calculated), typeof(string));
            outputsTable.Columns.Add(nameof(IInferredCrackPOFSCCCalculatorOutputs.ErrorMessage), typeof(string));

            outputsTable.Columns.Add(nameof(IInferredCrackPOFSCCCalculatorOutputs.FailureProbability), typeof(double));
        }

        /// <summary>
        /// Saves the input and outputs in passed data row.
        /// </summary>
        /// <param name="inputs">The inputs row.</param>
        /// <param name="outputs">The outputs.</param>
        /// <param name="outputsRow">The row.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputs"/>, <paramref name="outputs"/>, <paramref name="outputsRow"/> parameter is <b>null</b>.</exception>
        protected override void SaveInputOutputs(IInferredCrackPOFSCCCalculatorInputs inputs, IInferredCrackPOFSCCCalculatorOutputs outputs, DataRow outputsRow)
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
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.NumberOfIterations)] = inputs.NumberOfIterations;
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.NumberOfYears)] = inputs.NumberOfYears;
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.Id)] = inputs.Id;
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.RangeId)] = inputs.RangeId;
            if (inputs.NominalWallThickness_mm != null)
            {
                outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.NominalWallThickness_mm)] = inputs.NominalWallThickness_mm;
            }
            if (inputs.OutsideDiameter_in != null)
            {
                outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.OutsideDiameter_in)] = inputs.OutsideDiameter_in;
            }
            if (inputs.PipeGrade_MPa != null)
            {
                outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.PipeGrade_MPa)] = inputs.PipeGrade_MPa;
            }
            if (inputs.Toughness_Joule != null)
            {
                outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.Toughness_Joule)] = inputs.Toughness_Joule;
            }
            if (inputs.MaximumAllowableOperatingPressure_kPa != null)
            {
                outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.MaximumAllowableOperatingPressure_kPa)] = inputs.MaximumAllowableOperatingPressure_kPa;
            }
            if (inputs.ILIMeasuredClusterDefectLength_mm != null)
            {
                outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectLength_mm)] = inputs.ILIMeasuredClusterDefectLength_mm;
            }
            if (inputs.ILIMeasuredClusterDefectDepthPrc != null)
            {
                outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)] = inputs.ILIMeasuredClusterDefectDepthPrc;
            }
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.SurfaceIndicator)] = inputs.SurfaceIndicator;
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.InstallationDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.InstallationDate);
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorInputs.ILIDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.ILIDate);

            // create columns for common data
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorOutputs.Elapsed)] = outputs.Elapsed;
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorOutputs.Calculated)] = outputs.Calculated ? "Y" : "N";
            outputsRow[nameof(IInferredCrackPOFSCCCalculatorOutputs.ErrorMessage)] = outputs.ErrorMessage;

            outputsRow[nameof(IInferredCrackPOFSCCCalculatorOutputs.FailureProbability)] = outputs.FailureProbability;
        }
        #endregion
    }
}
