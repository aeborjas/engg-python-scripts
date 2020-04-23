using System;
using System.Collections.Concurrent;
using System.Data;
using System.Diagnostics;
using System.Threading.Tasks;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;

using Dynamic.MonteCarloSimulation.Properties;

namespace Dynamic.MonteCarloSimulation.CrackPOF_SCC
{
	/// <summary>
	/// Implements Crack POF, SCC calculator
	/// </summary>
	/// <seealso cref="Dynamic.MonteCarloSimulation.SimulationCalculator{Dynamic.MonteCarloSimulation.CrackPOF_SCC.ICrackPOFSCCCalculatorInputs, Dynamic.MonteCarloSimulation.CrackPOF_SCC.ICrackPOFSCCCalculatorOutputs, Dynamic.MonteCarloSimulation.ISimulationIntermediates}" />
	public class CrackPOFSCCCalculator
        : SimulationCalculator<ICrackPOFSCCCalculatorInputs, ICrackPOFSCCCalculatorOutputs, ICrackPOFSCCCalculatorIntermediates>
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
        private bool CanCalculate(ICrackPOFSCCCalculatorInputs inputs)
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
		protected override ICrackPOFSCCCalculatorOutputs CalculateSingleRow(
			ICrackPOFSCCCalculatorInputs inputs,
			ConcurrentBag<ICrackPOFSCCCalculatorIntermediates> intermediates = null
			)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            ICrackPOFSCCCalculatorOutputs outputs = new CrackPOFSCCCalculatorOutputs();

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
            double PeakDepth = inputs.ILIMeasuredClusterDefectDepthPrc != null ? (double)inputs.ILIMeasuredClusterDefectDepthPrc : 0;
            double pipeGrade = inputs.PipeGrade_MPa != null ? (double)inputs.PipeGrade_MPa : 0;
            double? pipeToughness = inputs.Toughness_Joule;
            double maop = inputs.MaximumAllowableOperatingPressure_kPa != null ? (double)inputs.MaximumAllowableOperatingPressure_kPa : 0;
            DateTime pipeInstallDate = inputs.InstallationDate.Value;
            DateTime iliDate = inputs.ILIDate.Value;

            string iliCompany = inputs.ILICompany;

            double? lengthGrowthRate = inputs.Crack_Growth_Rate;

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
            double CL_sd;
            if (IsRosen(iliCompany))
            {
                CL_sd = c_SCCCL_sd_Rosen;
            }
            else
            {
                CL_sd = c_SCCCL_sd_PII;
            }

            // Crack Depth, SD (mm)
            double CD_sd;
            if (IsRosen(iliCompany))
            {
                if (wt < 10)
                {
                    CD_sd = wt * c_SCCCD_sd_Rosen_wt_LT10;
                }
                else
                {
                    CD_sd = wt * c_SCCCD_sd_Rosen_wt_GT10;
                }
            }
            else
            {
                CD_sd = c_SCCCD_sd_PII;
            }

            // create instance of failure flags...
            var allFlags = new ConcurrentBag<double[,]>();

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
                    double CrackLength = Math.Max(1, Normal.InvCDF((double)ILIDefectL * c_CL_m, CL_sd, SystemRandomSource.Default.NextDouble())) * Convert_MM_to_Inch;

                    // Crack Depth 1 Growth Rate (in/yr)
                    double CrackDepthGR1 = (lengthGrowthRate != null ? (double)lengthGrowthRate : 0.1 * Math.Pow(-Math.Log(1 - SystemRandomSource.Default.NextDouble()), 1 / 2.55)) * Convert_MM_to_Inch;

                    // Crack Depth, Run (in)
                    double CrackDepthRun1 = Math.Max(0, Normal.InvCDF((double)PeakDepth * wt * c_CD_m, CD_sd, SystemRandomSource.Default.NextDouble())) * Convert_MM_to_Inch;

                    // Crack Depth (in)
                    double CrackDepth1 = CrackDepthRun1 + (DateTime.Now - iliDate).TotalDays / 365.25 * CrackDepthGR1;

                    // Elliptical C Equivalent Shape
                    double EllipticalC = Math.Min(Math.PI * CrackLength / 4, Math.Sqrt(67.2 * D_p * Wt_p / 2));

                    // CRT Ratio
                    double CRTRatio1 = Math.Min(93.753, Math.Pow(EllipticalC, 2) / (D_p * Wt_p / 2));

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

                    // if we have a list to store intermediates
                    if (intermediates != null)
                    {
                        intermediates.Add(new CrackPOFSCCCalculatorIntermediates()
                        {
                            Id = inputs.Id,
                            Young_p = Young_p,
                            S_p = S_p,
                            D_p = D_p,
                            Wt_p = Wt_p,
                            CrackLength = CrackLength,
                            CrackDepthGR1 = CrackDepthGR1,
                            CrackDepthRun1 = CrackDepthRun1,
                            CrackDepth1 = CrackDepth1,
                            EllipticalC = EllipticalC,
                            CRTRatio1 = CRTRatio1,
                            FlowStress1 = FlowStress1,
                            Y1 = Y1,
                            X1 = X1,
                            Mt1 = Mt1,
                            MP1 = MP1,
                            FailureStress1 = FailureStress1,
                            CrackFailPressure1 = CrackFailPressure1,
                            LeakFlag = leakFlag
                        });
                    }

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
                if (allFlags.TryTake(out double[,]  item))
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
        protected override ICrackPOFSCCCalculatorInputs LoadInputs(DataRow row)
        {
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            ICrackPOFSCCCalculatorInputs inputs = new CrackPOFSCCCalculatorInputs();

            inputs.NumberOfIterations = (int)row[nameof(ICrackPOFSCCCalculatorInputs.NumberOfIterations)];
            inputs.NumberOfYears = (int)row[nameof(ICrackPOFSCCCalculatorInputs.NumberOfYears)];
            inputs.Id = Convert.ToInt32(row[nameof(ICrackPOFSCCCalculatorInputs.Id)].ToString());
            inputs.RangeId = Convert.ToInt32(row[nameof(ICrackPOFSCCCalculatorInputs.RangeId)].ToString());
            inputs.NominalWallThickness_mm = row[nameof(ICrackPOFSCCCalculatorInputs.NominalWallThickness_mm)].IfNullable<double>();
            inputs.OutsideDiameter_in = row[nameof(ICrackPOFSCCCalculatorInputs.OutsideDiameter_in)].IfNullable<double>();
            inputs.PipeGrade_MPa = row[nameof(ICrackPOFSCCCalculatorInputs.PipeGrade_MPa)].IfNullable<double>();
            inputs.Toughness_Joule = row[nameof(ICrackPOFSCCCalculatorInputs.Toughness_Joule)].IfNullable<double>();
            inputs.MaximumAllowableOperatingPressure_kPa = row[nameof(ICrackPOFSCCCalculatorInputs.MaximumAllowableOperatingPressure_kPa)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectLength_mm = row[nameof(ICrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectLength_mm)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectDepthPrc = row[nameof(ICrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)].IfNullable<double>();
            inputs.SurfaceIndicator = row[nameof(ICrackPOFSCCCalculatorInputs.SurfaceIndicator)].ToString();
            inputs.Crack_Growth_Rate = row[nameof(ICrackPOFSCCCalculatorInputs.Crack_Growth_Rate)].IfNullable<double>();
            inputs.InstallationDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(ICrackPOFSCCCalculatorInputs.InstallationDate)].ToString());
            inputs.ILIDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(ICrackPOFSCCCalculatorInputs.ILIDate)].ToString());
            inputs.ILICompany = row[nameof(ICrackPOFSCCCalculatorInputs.ILICompany)].ToString();

            return inputs;
        }

        /// <summary>
        /// Creates the output table columns.
        /// </summary>
        /// <param name="inputsTable">The inputs table.</param>
        /// <param name="outputsTable">The outputs table.</param>
        /// <param name="outputs">The outputs.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputsTable" />, <paramref name="outputsTable" />, <paramref name="outputs"/> parameter is <b>null</b>.</exception>
        protected override void CreateOutputTableColumns(DataTable inputsTable, DataTable outputsTable, ICrackPOFSCCCalculatorOutputs outputs)
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
            outputsTable.Columns.Add(nameof(ICrackPOFSCCCalculatorOutputs.Elapsed), typeof(double));
            outputsTable.Columns.Add(nameof(ICrackPOFSCCCalculatorOutputs.Calculated), typeof(string));
            outputsTable.Columns.Add(nameof(ICrackPOFSCCCalculatorOutputs.ErrorMessage), typeof(string));

            outputsTable.Columns.Add(nameof(ICrackPOFSCCCalculatorOutputs.FailureProbability), typeof(double));
        }

        /// <summary>
        /// Saves the input and outputs in passed data row.
        /// </summary>
        /// <param name="inputs">The inputs row.</param>
        /// <param name="outputs">The outputs.</param>
        /// <param name="outputsRow">The row.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputs"/>, <paramref name="outputs"/>, <paramref name="outputsRow"/> parameter is <b>null</b>.</exception>
        protected override void SaveInputOutputs(ICrackPOFSCCCalculatorInputs inputs, ICrackPOFSCCCalculatorOutputs outputs, DataRow outputsRow)
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
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.NumberOfIterations)] = inputs.NumberOfIterations;
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.NumberOfYears)] = inputs.NumberOfYears;
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.Id)] = inputs.Id;
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.RangeId)] = inputs.RangeId;
            if (inputs.NominalWallThickness_mm != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.NominalWallThickness_mm)] = inputs.NominalWallThickness_mm;
            }
            if (inputs.OutsideDiameter_in != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.OutsideDiameter_in)] = inputs.OutsideDiameter_in;
            }
            if (inputs.PipeGrade_MPa != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.PipeGrade_MPa)] = inputs.PipeGrade_MPa;
            }
            if (inputs.Toughness_Joule != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.Toughness_Joule)] = inputs.Toughness_Joule;
            }
            if (inputs.MaximumAllowableOperatingPressure_kPa != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.MaximumAllowableOperatingPressure_kPa)] = inputs.MaximumAllowableOperatingPressure_kPa;
            }
            if (inputs.ILIMeasuredClusterDefectLength_mm != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectLength_mm)] = inputs.ILIMeasuredClusterDefectLength_mm;
            }
            if (inputs.ILIMeasuredClusterDefectDepthPrc != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)] = inputs.ILIMeasuredClusterDefectDepthPrc;
            }
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.SurfaceIndicator)] = inputs.SurfaceIndicator;
            if (inputs.Crack_Growth_Rate != null)
            {
                outputsRow[nameof(ICrackPOFSCCCalculatorInputs.Crack_Growth_Rate)] = inputs.Crack_Growth_Rate;
            }
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.InstallationDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.InstallationDate);
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.ILIDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.ILIDate);
            outputsRow[nameof(ICrackPOFSCCCalculatorInputs.ILICompany)] = inputs.ILICompany;

            // create columns for common data
            outputsRow[nameof(ICrackPOFSCCCalculatorOutputs.Elapsed)] = outputs.Elapsed;
            outputsRow[nameof(ICrackPOFSCCCalculatorOutputs.Calculated)] = outputs.Calculated ? "Y" : "N";
            outputsRow[nameof(ICrackPOFSCCCalculatorOutputs.ErrorMessage)] = outputs.ErrorMessage;

            outputsRow[nameof(ICrackPOFSCCCalculatorOutputs.FailureProbability)] = outputs.FailureProbability;
        }
        #endregion
    }
}
