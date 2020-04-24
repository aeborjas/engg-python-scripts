using System;
using System.Collections.Concurrent;
using System.Data;
using System.Diagnostics;
using System.Threading.Tasks;
using Dynamic.MonteCarloSimulation.Properties;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;

namespace Dynamic.MonteCarloSimulation.Inferred_CrackPOF_CSCC
{
	/// <summary>
	/// Implements Inferred Circumferential Crack POF, SCC calculator
	/// </summary>
	/// <seealso cref="Dynamic.MonteCarloSimulation.SimulationCalculator{Dynamic.MonteCarloSimulation.Inferred_CrackPOF_CSCC.IInferredCrackPOFCSCCCalculatorInputs, Dynamic.MonteCarloSimulation.Inferred_CrackPOF_CSCC.IInferredCrackPOFCSCCCalculatorOutputs, Dynamic.MonteCarloSimulation.ISimulationIntermediates}" />
	public class InferredCrackPOFCSCCCalculator
        : SimulationCalculator<IInferredCrackPOFCSCCCalculatorInputs, IInferredCrackPOFCSCCCalculatorOutputs, ISimulationIntermediates>
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
        // SCC Crack Width, sd, Rosen(mm)
        private const double c_SCCCW_sd_Rosen = 7.80;
        // SCC Crack Width, sd, PII(mm)
        private const double c_SCCCW_sd_PII = 6.1;
        // SCC Crack Width, mean (fraction)
        private const double c_CW_m = 1.0;
        // SCC Crack Depth, sd, Rosen, (mm)
        private const double c_SCCCD_sd_Rosen = 0.78;
        // SCC Crack Depth, sd, PII, (mm)
        private const double c_SCCCD_sd_PII = 0.31;
        // SCC Crack Depth, mean(fraction)
        private const double c_CD_m = 1.00;
        // Inferred Crack Depth, sd, (mm)
        private const double c_CD_sd_default = 0.310;
        // Inferred Crack Length, sd, (mm)
        private const double c_CW_sd_default = 6.1;

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
        private bool CanCalculate(IInferredCrackPOFCSCCCalculatorInputs inputs)
        {
            // certain values must be present
            return (inputs.ILIDate != null) &&
                   (inputs.NominalWallThickness_mm != null) &&
                   (inputs.OutsideDiameter_in != null);
        }

		/// <summary>
		/// Calculates the single row.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="intermediates">The target to store intermediates in simulation. Optional, if not passed, then no intermediates will be stored.</param>
		/// <returns>
		/// Instance of <see cref="IInferredCrackPOFCSCCCalculatorOutputs" /> with calculation results.
		/// </returns>
		/// <exception cref="System.ArgumentNullException">Thrown when <paramref name="inputs" /> parameter is <b>null</b>.</exception>
		protected override IInferredCrackPOFCSCCCalculatorOutputs CalculateSingleRow(
			IInferredCrackPOFCSCCCalculatorInputs inputs,
			ConcurrentBag<ISimulationIntermediates> intermediates = null
			)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            IInferredCrackPOFCSCCCalculatorOutputs outputs = new InferredCrackPOFCSCCCalculatorOutputs();

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

            // Crack Depth, SD (mm)
            double CD_sd = c_CD_sd_default;

            // Crack Width, SD (mm)
            double CW_sd = c_CW_sd_default;

            // create instance of failure flags...
            var allFlags = new ConcurrentBag<double[,]>();

            //  This Parallel.For() construct accumulates counts in the TFailureFlags class and periodically (but not every
            //  iteration) updates the flags instance from the threadflags instance and creates a new threadflags instance.
            Parallel.For(0, numberOfIterations,
                () => new double[FlagsArrayCount, 1],
                (j, loop, threadflags) =>
                {
                    // Crack Depth Growth Rate (mm/yr)
                    double CrackDepthGR = 0.26 * Math.Sqrt(-Math.Log(1 - SystemRandomSource.Default.NextDouble()));

                    // Crack Width Growth Rate(mm / yr)
                    double CrackWidthGR = 3.1264 * Math.Pow(-Math.Log(1 - SystemRandomSource.Default.NextDouble()), (1 / 1.8742));

                    // Outside Diameter, distributed (mm)
                    double D_p = Normal.InvCDF(
                        od * c_D_m,
                        od * c_D_sd,
						SystemRandomSource.Default.NextDouble()) * Convert_Inch_To_MM;

                    // Wall thickness, distributed (mm)
                    double Wt_p = Normal.InvCDF(
                        wt * c_Wt_m,
                        wt * c_Wt_sd,
						SystemRandomSource.Default.NextDouble());

                    // Crack Width, Run (mm)
                    double CrackWidthRun = Math.Max(1, Normal.InvCDF((double)ILIDefectL * c_CW_m, CW_sd, SystemRandomSource.Default.NextDouble()));

                    // Crack Width (mm)
                    double CrackWidth = CrackWidthRun + (DateTime.Now - iliDate).TotalDays * CrackWidthGR;

                    // Crack Depth, Run (mm)
                    double CrackDepthRun = Math.Max(0, Normal.InvCDF((double)peakDepth * wt * c_CD_m, CD_sd, SystemRandomSource.Default.NextDouble()));

                    // Crack Depth (mm)
                    double CrackDepth = CrackDepthRun + (DateTime.Now - iliDate).TotalDays / 365.25 * CrackDepthGR;

                    // Leak flag
                    // For a purpose of this algorithm - treat everything as leaks
                    double factor_1 = 0.40 * Math.PI * D_p;
                    double factor_2 = 0.4 * Wt_p;
                    double factor_3 = 0.6 * Wt_p;
                    int leakFlag = (
                                    ((CrackWidth >= factor_1) &&
                                     (CrackDepth >= factor_2)) ||
                                    (CrackDepth >= factor_3)
                                   ) ? 1 : 0;

                    // Increment cumulative flags
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
        protected override IInferredCrackPOFCSCCCalculatorInputs LoadInputs(DataRow row)
        {
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            IInferredCrackPOFCSCCCalculatorInputs inputs = new InferredCrackPOFCSCCCalculatorInputs();

            inputs.NumberOfIterations = (int)row[nameof(IInferredCrackPOFCSCCCalculatorInputs.NumberOfIterations)];
            inputs.NumberOfYears = (int)row[nameof(IInferredCrackPOFCSCCCalculatorInputs.NumberOfYears)];
            inputs.Id = Convert.ToInt32(row[nameof(IInferredCrackPOFCSCCCalculatorInputs.Id)].ToString());
            inputs.RangeId = Convert.ToInt32(row[nameof(IInferredCrackPOFCSCCCalculatorInputs.RangeId)].ToString());
            inputs.NominalWallThickness_mm = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.NominalWallThickness_mm)].IfNullable<double>();
            inputs.OutsideDiameter_in = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.OutsideDiameter_in)].IfNullable<double>();
            inputs.PipeGrade_MPa = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.PipeGrade_MPa)].IfNullable<double>();
            inputs.Toughness_Joule = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.Toughness_Joule)].IfNullable<double>();
            inputs.MaximumAllowableOperatingPressure_kPa = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.MaximumAllowableOperatingPressure_kPa)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectLength_mm = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.ILIMeasuredClusterDefectLength_mm)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectDepthPrc = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)].IfNullable<double>();
            inputs.SurfaceIndicator = row[nameof(IInferredCrackPOFCSCCCalculatorInputs.SurfaceIndicator)].ToString();
            inputs.InstallationDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(IInferredCrackPOFCSCCCalculatorInputs.InstallationDate)].ToString());
            inputs.ILIDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(IInferredCrackPOFCSCCCalculatorInputs.ILIDate)].ToString());

            return inputs;
        }

        /// <summary>
        /// Creates the output table columns.
        /// </summary>
        /// <param name="inputsTable">The inputs table.</param>
        /// <param name="outputsTable">The outputs table.</param>
        /// <param name="outputs">The outputs.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputsTable" />, <paramref name="outputsTable" />, <paramref name="outputs"/> parameter is <b>null</b>.</exception>
        protected override void CreateOutputTableColumns(DataTable inputsTable, DataTable outputsTable, IInferredCrackPOFCSCCCalculatorOutputs outputs)
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
            outputsTable.Columns.Add(nameof(IInferredCrackPOFCSCCCalculatorOutputs.Elapsed), typeof(double));
            outputsTable.Columns.Add(nameof(IInferredCrackPOFCSCCCalculatorOutputs.Calculated), typeof(string));
            outputsTable.Columns.Add(nameof(IInferredCrackPOFCSCCCalculatorOutputs.ErrorMessage), typeof(string));

            outputsTable.Columns.Add(nameof(IInferredCrackPOFCSCCCalculatorOutputs.FailureProbability), typeof(double));
        }

        /// <summary>
        /// Saves the input and outputs in passed data row.
        /// </summary>
        /// <param name="inputs">The inputs row.</param>
        /// <param name="outputs">The outputs.</param>
        /// <param name="outputsRow">The row.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputs"/>, <paramref name="outputs"/>, <paramref name="outputsRow"/> parameter is <b>null</b>.</exception>
        protected override void SaveInputOutputs(IInferredCrackPOFCSCCCalculatorInputs inputs, IInferredCrackPOFCSCCCalculatorOutputs outputs, DataRow outputsRow)
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
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.NumberOfIterations)] = inputs.NumberOfIterations;
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.NumberOfYears)] = inputs.NumberOfYears;
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.Id)] = inputs.Id;
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.RangeId)] = inputs.RangeId;
            if (inputs.NominalWallThickness_mm != null)
            {
                outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.NominalWallThickness_mm)] = inputs.NominalWallThickness_mm;
            }
            if (inputs.OutsideDiameter_in != null)
            {
                outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.OutsideDiameter_in)] = inputs.OutsideDiameter_in;
            }
            if (inputs.PipeGrade_MPa != null)
            {
                outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.PipeGrade_MPa)] = inputs.PipeGrade_MPa;
            }
            if (inputs.Toughness_Joule != null)
            {
                outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.Toughness_Joule)] = inputs.Toughness_Joule;
            }
            if (inputs.MaximumAllowableOperatingPressure_kPa != null)
            {
                outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.MaximumAllowableOperatingPressure_kPa)] = inputs.MaximumAllowableOperatingPressure_kPa;
            }
            if (inputs.ILIMeasuredClusterDefectLength_mm != null)
            {
                outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.ILIMeasuredClusterDefectLength_mm)] = inputs.ILIMeasuredClusterDefectLength_mm;
            }
            if (inputs.ILIMeasuredClusterDefectDepthPrc != null)
            {
                outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)] = inputs.ILIMeasuredClusterDefectDepthPrc;
            }
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.SurfaceIndicator)] = inputs.SurfaceIndicator;
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.InstallationDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.InstallationDate);
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorInputs.ILIDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.ILIDate);

            // create columns for common data
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorOutputs.Elapsed)] = outputs.Elapsed;
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorOutputs.Calculated)] = outputs.Calculated ? "Y" : "N";
            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorOutputs.ErrorMessage)] = outputs.ErrorMessage;

            outputsRow[nameof(IInferredCrackPOFCSCCCalculatorOutputs.FailureProbability)] = outputs.FailureProbability;
        }
        #endregion
    }
}
