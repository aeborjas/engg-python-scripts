using System;
using System.Collections.Concurrent;
using System.Data;
using System.Diagnostics;
using System.Threading.Tasks;

using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;

using Dynamic.MonteCarloSimulation.Properties;

namespace Dynamic.MonteCarloSimulation.TypeE
{
	/// <summary>
	/// Implements Type E Simulation calculator
	/// </summary>
	/// <seealso cref="Dynamic.MonteCarloSimulation.SimulationCalculator{Dynamic.MonteCarloSimulation.TypeE.ITypeECalculatorInputs, Dynamic.MonteCarloSimulation.TypeE.ITypeECalculatorOutputs, Dynamic.MonteCarloSimulation.ISimulationIntermediates}" />
	public class TypeECalculator
        : SimulationCalculator<ITypeECalculatorInputs, ITypeECalculatorOutputs, ISimulationIntermediates>
    {
        #region Constants
        // UTS, mean
        private const double S_m = 1.09;

        // UTS, sd
        private const double S_sd = 0.044;

        // wall thickness, mean
        private const double Wt_m = 1.01;

        // wall thickness, sd
        private const double Wt_sd = 0.01;

        // ILI Defect Length, sd
        private const double ILIDefectL_sd = 0.61;

        // ILI Defect Depth, sd
        private const double ILIDefectD_sd = 0.078;

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
        private bool CanCalculate(ITypeECalculatorInputs inputs)
        {
            return
                (inputs.NominalWallThickness_mm != null) &&
                (inputs.OutsideDiameter_in != null) &&
                (inputs.ILIMeasuredClusterDefectLength_mm != null) &&
                (inputs.ILIMeasuredClusterDefectDepthPrc != null) &&
                (inputs.PipeGrade_MPa != null) &&
                (inputs.MaximumAllowableOperatingPressure_kPa != null) &&
                (inputs.InstallationDate != null) &&
                (inputs.ILIDate != null);
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
		protected override ITypeECalculatorOutputs CalculateSingleRow(
			ITypeECalculatorInputs inputs,
			ConcurrentBag<ISimulationIntermediates> intermediates = null
			)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            ITypeECalculatorOutputs outputs = new TypeECalculatorOutputs();

            // do up front check if we can calculate on this inputs
            if (!CanCalculate(inputs))
            {
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
            double maop = inputs.MaximumAllowableOperatingPressure_kPa != null ? (double)inputs.MaximumAllowableOperatingPressure_kPa : 0;
            DateTime pipeInstallDate = inputs.InstallationDate.Value;
            DateTime iliDate = inputs.ILIDate.Value;

            int? growthStartYear = inputs.YearAssumedForStartOfCorrosionGrowth;
            bool allowLengthGrowth = string.Equals(inputs.AllowFlawGrowthInLengthDirection ?? "N", "Y", StringComparison.CurrentCultureIgnoreCase);

            double? depthGrowthRate = string.Equals(inputs.SurfaceIndicator, "I", StringComparison.CurrentCultureIgnoreCase) ? inputs.Growth_Rate_Depth_Internal : inputs.Growth_Rate_Depth_External;
            double? lengthGrowthRate = string.Equals(inputs.SurfaceIndicator, "I", StringComparison.CurrentCultureIgnoreCase) ? inputs.Growth_Rate_Length_Internal : inputs.Growth_Rate_Length_External;

            // Corrosion Growth Start Year
            int corrosionGrowthStartYr = growthStartYear != null ? (int)growthStartYear : (pipeInstallDate.Year + iliDate.Year) / 2;

            // ILI Growth Period (yrs)
            int ILIGrowthPeriod = DateTime.Now.Year - corrosionGrowthStartYr;

            // Growth Time (yrs)
            double growthTime = (DateTime.Now - iliDate).TotalDays / 365.25;

            // MeasuredDefectDepth (in)
            double MeasuredDefectDepth = wt * Convert_MM_to_Inch * PeakDepth;

            // Flow Stress, nominal (PSI)
            double FlowStress = pipeGrade * Convert_MPa_PSI + 10000;

            // l2Dt factor, nominal
            double l2Dt = Math.Pow(ILIDefectL * Convert_MM_to_Inch, 2) / (od * wt * Convert_MM_to_Inch);

            // Folias Factor, nominal
            double FoliasFactor;
            if (l2Dt <= 50)
            {
                FoliasFactor = Math.Sqrt(1 + 0.6275 * l2Dt - 0.003375 * Math.Pow(l2Dt, 2));
            }
            else
            {
                FoliasFactor = 0.032 * l2Dt + 3.3;
            }

            // Failure Pressure, nominal (PSI)
            double FailurePressure = FlowStress * (1 - 0.85 * (MeasuredDefectDepth / (wt * Convert_MM_to_Inch))) / (1 - 0.85 * (MeasuredDefectDepth / (wt * Convert_MM_to_Inch * FoliasFactor)));

            // pass some calculated values back
            outputs.Growth_Rate_Depth = depthGrowthRate;
            outputs.Growth_Rate_Length = lengthGrowthRate;

            // create instance of failure flags...
            var allFlags = new ConcurrentBag<double[,]>();

            //  This Parallel.For() construct accumulates counts in the TFailureFlags class and periodically (but not every
            //  iteration) updates the flags instance from the threadflags instance and creates a new threadflags instance.
            Parallel.For(0, numberOfIterations,
                () => new double[FlagsArrayCount, 1],
                (j, loop, threadflags) =>
                {
                    double S_p = -1;
                    double Wt_p = -1;
                    double ILIDefectL_ILI = -1;
                    double DefectLengthGR = -1;
                    double ILIDefectL_Current = -1;
                    double ILIDefectD_ILI = -1;
                    double DefectDepthGR = -1;
                    double ILIDefectD_Current = -1;
                    double FlowStress_p = -1;
                    double OpStress_p = -1;
                    double l2Dt_p = -1;
                    double FoliasFactor_p = -1;
                    double ModelError = -1;
                    double FailurePressure_p = -1;
                    int ruptureFlag = -1;
                    int leakFlag = -1;
                    string errorMessage = string.Empty;

                    try
                    {
                        //  Get randomized input variables, statistically near the original values.

                        // Grade, distributed (MPa)
                        S_p = Normal.InvCDF(
                            pipeGrade * S_m,
                            pipeGrade * S_sd,
							SystemRandomSource.Default.NextDouble());

                        // Wall thickness, distributed (mm)
                        Wt_p = Math.Max(0, Normal.InvCDF(
                            wt * Wt_m,
                            wt * Wt_sd,
							SystemRandomSource.Default.NextDouble()));

                        // ILI Defect Length in ILIYear (in)
                        ILIDefectL_ILI = Math.Max(0, Normal.InvCDF(
                            ILIDefectL * Convert_MM_to_Inch,
                            ILIDefectL_sd, SystemRandomSource.Default.NextDouble()));

                        // Defect Length Growth Rate (in/yr)
                        DefectLengthGR = 0;
                        if (allowLengthGrowth)
                        {
                            DefectLengthGR = lengthGrowthRate != null ? (double)lengthGrowthRate * Convert_MM_to_Inch : ILIDefectL_ILI / ILIGrowthPeriod;
                        }

                        // ILI Defect Length Current (in)
                        ILIDefectL_Current = ILIDefectL_ILI + growthTime * DefectLengthGR;

                        // ILI Defect Depth in ILIYear (in)
                        ILIDefectD_ILI = Math.Max(0, Normal.InvCDF(
                            MeasuredDefectDepth,
                            wt * Convert_MM_to_Inch * ILIDefectD_sd,
							SystemRandomSource.Default.NextDouble()));

                        // Defect Depth Growth Rate (in/yr)
                        if (depthGrowthRate != null)
                        {
                            DefectDepthGR = (double)depthGrowthRate;
                        }
                        else
                        {
                            DefectDepthGR = 0.1 * Math.Pow(-Math.Log(1 - SystemRandomSource.Default.NextDouble()), (1 / 1.439)) * Convert_MM_to_Inch;
                        }

                        // ILI Defect Depth Current (in)
                        ILIDefectD_Current = ILIDefectD_ILI + growthTime * DefectDepthGR;

                        // Flow Stress, distributed (PSI)
                        FlowStress_p = S_p * Convert_MPa_PSI + 10000;

                        // Operating Stress, distributed (PSI)
                        OpStress_p = od * maop * Convert_kPa_PSI / (2 * Wt_p * Convert_MM_to_Inch);

                        // l2Dt factor, distributed
                        l2Dt_p = Math.Pow(ILIDefectL_Current, 2) / (od * Wt_p * Convert_MM_to_Inch);

                        // Folias Factor, distributed
                        if (l2Dt_p <= 50)
                        {
                            FoliasFactor_p = Math.Sqrt((1 + 0.6275 * l2Dt_p - 0.003375 * Math.Pow(l2Dt_p, 2)));
                        }
                        else
                        {
                            FoliasFactor_p = 0.032 * l2Dt_p + 3.3;
                        }

                        // Model Error
                        ModelError = 0.914 + Statistic.GAMMAINV_Random_2175_0225();

                        // Failure Pressure, distributed (PSI)
                        FailurePressure_p = (FlowStress_p * ((1 - 0.85 * (ILIDefectD_Current / (Wt_p * Convert_MM_to_Inch))) / (1 - 0.85 * (ILIDefectD_Current / (Wt_p * Convert_MM_to_Inch * FoliasFactor_p))))) * ModelError;

                        // Rupture flag
                        ruptureFlag = (OpStress_p >= FailurePressure_p) ? 1 : 0;

                        // Leak flag
                        leakFlag = (ILIDefectD_Current >= 0.8 * wt * Convert_MM_to_Inch) ? 1 : 0;

                        //  Increment cumulative flags
                        threadflags[LeakFlags_index, 0] += leakFlag;
                        threadflags[RuptureFlags_index, 0] += ruptureFlag;
                        threadflags[LeakOrRuptureFlags_index, 0] += Math.Max(ruptureFlag, leakFlag); 
                        threadflags[Iterations_index, 0] += 1;  //  Sanity check that we're not losing results due to Parallel.
                    }
                    catch (Exception e)
                    {
                        errorMessage = e.Message;
                    }

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

            //  Convert flags to probabilities, and determine failures
            // don't forget to convert it in DOUBLE...
            double poeLeak = flags[LeakFlags_index, 0] * 1.0 / flags[Iterations_index, 0];
            outputs.POE_Leak = poeLeak;
            double poeRupture = flags[RuptureFlags_index, 0] * 1.0 / flags[Iterations_index, 0];
            outputs.POE_Rupture = poeRupture;

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
        protected override ITypeECalculatorInputs LoadInputs(DataRow row)
        {
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            ITypeECalculatorInputs inputs = new TypeECalculatorInputs();

            inputs.NumberOfIterations = (int)row[nameof(ITypeECalculatorInputs.NumberOfIterations)];
            inputs.NumberOfYears = (int)row[nameof(ITypeECalculatorInputs.NumberOfYears)];
            inputs.Id = Convert.ToInt32(row[nameof(ITypeECalculatorInputs.Id)].ToString());
            inputs.RangeId = Convert.ToInt32(row[nameof(ITypeECalculatorInputs.RangeId)].ToString());
            inputs.NominalWallThickness_mm = row[nameof(ITypeECalculatorInputs.NominalWallThickness_mm)].IfNullable<double>();
            inputs.OutsideDiameter_in = row[nameof(ITypeECalculatorInputs.OutsideDiameter_in)].IfNullable<double>();
            inputs.PipeGrade_MPa = row[nameof(ITypeECalculatorInputs.PipeGrade_MPa)].IfNullable<double>();
            inputs.MaximumAllowableOperatingPressure_kPa = row[nameof(ITypeECalculatorInputs.MaximumAllowableOperatingPressure_kPa)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectLength_mm = row[nameof(ITypeECalculatorInputs.ILIMeasuredClusterDefectLength_mm)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectDepthPrc = row[nameof(ITypeECalculatorInputs.ILIMeasuredClusterDefectDepthPrc)].IfNullable<double>();
            inputs.YearAssumedForStartOfCorrosionGrowth = row[nameof(ITypeECalculatorInputs.YearAssumedForStartOfCorrosionGrowth)].IfNullable<int>();
            inputs.AllowFlawGrowthInLengthDirection = row[nameof(ITypeECalculatorInputs.AllowFlawGrowthInLengthDirection)].ToString();
            inputs.SurfaceIndicator = row[nameof(ITypeECalculatorInputs.SurfaceIndicator)].ToString();
            inputs.Growth_Rate_Depth_External = row[nameof(ITypeECalculatorInputs.Growth_Rate_Depth_External)].IfNullable<double>();
            inputs.Growth_Rate_Length_External = row[nameof(ITypeECalculatorInputs.Growth_Rate_Length_External)].IfNullable<double>();
            inputs.Growth_Rate_Depth_Internal = row[nameof(ITypeECalculatorInputs.Growth_Rate_Depth_Internal)].IfNullable<double>();
            inputs.Growth_Rate_Length_Internal = row[nameof(ITypeECalculatorInputs.Growth_Rate_Length_Internal)].IfNullable<double>();
            inputs.InstallationDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(ITypeECalculatorInputs.InstallationDate)].ToString());
            inputs.ILIDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(ITypeECalculatorInputs.ILIDate)].ToString());

            return inputs;
        }

        /// <summary>
        /// Creates the output table columns.
        /// </summary>
        /// <param name="inputsTable">The inputs table.</param>
        /// <param name="outputsTable">The outputs table.</param>
        /// <param name="outputs">The outputs.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputsTable" />, <paramref name="outputsTable" /> parameter is <b>null</b>.</exception>
        protected override void CreateOutputTableColumns(DataTable inputsTable, DataTable outputsTable, ITypeECalculatorOutputs outputs)
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
            outputsTable.Columns.Add(nameof(ITypeECalculatorOutputs.Elapsed), typeof(double));
            outputsTable.Columns.Add(nameof(ITypeECalculatorOutputs.Calculated), typeof(string));
            outputsTable.Columns.Add(nameof(ITypeECalculatorOutputs.ErrorMessage), typeof(string));

            outputsTable.Columns.Add(nameof(ITypeECalculatorOutputs.POE_Leak), typeof(double));
            outputsTable.Columns.Add(nameof(ITypeECalculatorOutputs.POE_Rupture), typeof(double));
            outputsTable.Columns.Add(nameof(ITypeECalculatorOutputs.Growth_Rate_Depth), typeof(double));
            outputsTable.Columns.Add(nameof(ITypeECalculatorOutputs.Growth_Rate_Length), typeof(double));
        }

        /// <summary>
        /// Saves the input and outputs in passed data row.
        /// </summary>
        /// <param name="inputs">The inputs row.</param>
        /// <param name="outputs">The outputs.</param>
        /// <param name="outputsRow">The row.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputs"/>, <paramref name="outputs"/>, <paramref name="outputsRow"/> parameter is <b>null</b>.</exception>
        protected override void SaveInputOutputs(ITypeECalculatorInputs inputs, ITypeECalculatorOutputs outputs, DataRow outputsRow)
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
            outputsRow[nameof(ITypeECalculatorInputs.NumberOfIterations)] = inputs.NumberOfIterations;
            outputsRow[nameof(ITypeECalculatorInputs.NumberOfYears)] = inputs.NumberOfYears;
            outputsRow[nameof(ITypeECalculatorInputs.Id)] = inputs.Id;
            outputsRow[nameof(ITypeECalculatorInputs.RangeId)] = inputs.RangeId;
            if (inputs.NominalWallThickness_mm != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.NominalWallThickness_mm)] = inputs.NominalWallThickness_mm;
            }
            if (inputs.OutsideDiameter_in != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.OutsideDiameter_in)] = inputs.OutsideDiameter_in;
            }
            if (inputs.PipeGrade_MPa != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.PipeGrade_MPa)] = inputs.PipeGrade_MPa;
            }
            if (inputs.MaximumAllowableOperatingPressure_kPa != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.MaximumAllowableOperatingPressure_kPa)] = inputs.MaximumAllowableOperatingPressure_kPa;
            }
            if (inputs.ILIMeasuredClusterDefectLength_mm != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.ILIMeasuredClusterDefectLength_mm)] = inputs.ILIMeasuredClusterDefectLength_mm;
            }
            if (inputs.ILIMeasuredClusterDefectDepthPrc != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.ILIMeasuredClusterDefectDepthPrc)] = inputs.ILIMeasuredClusterDefectDepthPrc;
            }
            if (inputs.YearAssumedForStartOfCorrosionGrowth != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.YearAssumedForStartOfCorrosionGrowth)] = inputs.YearAssumedForStartOfCorrosionGrowth;
            }
            if (inputs.AllowFlawGrowthInLengthDirection != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.AllowFlawGrowthInLengthDirection)] = inputs.AllowFlawGrowthInLengthDirection;
            }
            outputsRow[nameof(ITypeECalculatorInputs.SurfaceIndicator)] = inputs.SurfaceIndicator;
            if (inputs.Growth_Rate_Depth_External != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.Growth_Rate_Depth_External)] = inputs.Growth_Rate_Depth_External;
            }
            if (inputs.Growth_Rate_Length_External != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.Growth_Rate_Length_External)] = inputs.Growth_Rate_Length_External;
            }
            if (inputs.Growth_Rate_Depth_Internal != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.Growth_Rate_Depth_Internal)] = inputs.Growth_Rate_Depth_Internal;
            }
            if (inputs.Growth_Rate_Length_Internal != null)
            {
                outputsRow[nameof(ITypeECalculatorInputs.Growth_Rate_Length_Internal)] = inputs.Growth_Rate_Length_Internal;
            }
            outputsRow[nameof(ITypeECalculatorInputs.InstallationDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.InstallationDate);
            outputsRow[nameof(ITypeECalculatorInputs.ILIDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.ILIDate);

            // create columns for common data
            outputsRow[nameof(ITypeECalculatorOutputs.Elapsed)] = outputs.Elapsed;
            outputsRow[nameof(ITypeECalculatorOutputs.Calculated)] = outputs.Calculated ? "Y" : "N";
            outputsRow[nameof(ITypeECalculatorOutputs.ErrorMessage)] = outputs.ErrorMessage;

            outputsRow[nameof(ITypeECalculatorOutputs.POE_Leak)] = outputs.POE_Leak;
            outputsRow[nameof(ITypeECalculatorOutputs.POE_Rupture)] = outputs.POE_Rupture;
            if (outputs.Growth_Rate_Depth != null)
            {
                outputsRow[nameof(ITypeECalculatorOutputs.Growth_Rate_Depth)] = outputs.Growth_Rate_Depth;
            }
            if (outputs.Growth_Rate_Length != null)
            {
                outputsRow[nameof(ITypeECalculatorOutputs.Growth_Rate_Length)] = outputs.Growth_Rate_Length;
            }
        }
        #endregion
    }
}
