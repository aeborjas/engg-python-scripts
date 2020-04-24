using System;
using System.Collections.Concurrent;
using System.Data;
using System.Diagnostics;
using System.Threading.Tasks;

using MathNet.Numerics.Distributions;

using Dynamic.MonteCarloSimulation.Properties;
using MathNet.Numerics.Random;

namespace Dynamic.MonteCarloSimulation.MechanicalDamageResident
{
	/// <summary>
	/// Implements Mechanical Damage, Resident calculator
	/// </summary>
	/// <seealso cref="Dynamic.MonteCarloSimulation.SimulationCalculator{Dynamic.MonteCarloSimulation.MechanicalDamageResident.IMechanicalDamageResidentCalculatorInputs, Dynamic.MonteCarloSimulation.MechanicalDamageResident.IMechanicalDamageResidentCalculatorOutputs, Dynamic.MonteCarloSimulation.ISimulationIntermediates}" />
	public class MechanicalDamageResidentCalculator
        : SimulationCalculator<IMechanicalDamageResidentCalculatorInputs, IMechanicalDamageResidentCalculatorOutputs, ISimulationIntermediates>
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
        // UTS, mean
        private const double c_UTS_m = 1.12;
        // UTS, sd
        private const double c_UTS_sd = 0.035;
        // Dent Length, mean
        private const double c_DentL_m = 1;
        // Dent Width, mean
        private const double c_DentW_m = 1;
        // Dent Depth, mean
        private const double c_DentD_m = 1;
        // Gouge Depth, mean
        private const double c_GougeD_m = 1;
        // Gouge Depth, sd (fraction)
        private const double c_GougeD_sd = 0.078;
        // Dent Depth, standard deviation, BJ (mm)
        private const double c_DentD_sd_BJ = 2.5;
        // Dent Length, standard deviation, BJ (mm)
        private const double c_DentL_sd_BJ = 2.5;
        // Dent Width, standard deviation, BJ (mm)
        private const double c_DentW_sd_BJ = 2.5;
        // Dent Length, standard deviation, Rosen (mm)
        private const double c_DentL_sd_Rosen = 19.5;
        // Dent Width, standard deviation, Rosen (mm)
        private const double c_DentW_sd_Rosen = 39;
        // Dent Depth, standard deviation, TDW (fraction)
        private const double c_DentD_sdP_TDW = 0.0056;
        // Dent Length, standard deviation, TDW (mm)
        private const double c_DentL_sd_TDW = 7.6;
        // Dent Width, standard deviation, TDW (mm)
        private const double c_DentW_sd_TDW = 25.4;

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
        private bool CanCalculate(IMechanicalDamageResidentCalculatorInputs inputs)
        {
            // certain values must be present
            return (inputs.InstallationDate != null) &&
                (inputs.ILIDate != null) &&
                (inputs.PipeGrade_MPa != null) &&
                (inputs.MaximumAllowableOperatingPressure_kPa != null) &&
                (inputs.ILIMeasuredClusterDefectLength_mm != null) &&
                (inputs.ILIMeasuredClusterDefectWidth_mm != null) &&
                (inputs.OutsideDiameter_in != null) &&
                (inputs.NominalWallThickness_mm != null);
        }

		/// <summary>
		/// Calculates the single row.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="intermediates">The target to store intermediates in simulation. Optional, if not passed, then no intermediates will be stored.</param>
		/// <returns>
		/// Class with output results.
		/// </returns>
		/// <exception cref="System.ArgumentNullException">inputs</exception>
		protected override IMechanicalDamageResidentCalculatorOutputs CalculateSingleRow(
			IMechanicalDamageResidentCalculatorInputs inputs,
			ConcurrentBag<ISimulationIntermediates> intermediates = null
			)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            IMechanicalDamageResidentCalculatorOutputs outputs = new MechanicalDamageResidentCalculatorOutputs();

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

            // following values checked for null in CanCalculate method, so it's safe to just obtain values.
            double wt = inputs.NominalWallThickness_mm.Value;
            double od = inputs.OutsideDiameter_in.Value;
            double dentLength = inputs.ILIMeasuredClusterDefectLength_mm.Value;
            double dentWidth = inputs.ILIMeasuredClusterDefectWidth_mm.Value;

            double? gougeDepth = inputs.ILIMeasuredClusterDefectDepthPrc;
            double? dentDepth = inputs.ILIMeasuredClusterDefectDepthPrcNPS;

            double pipeGrade = inputs.PipeGrade_MPa != null ? (double)inputs.PipeGrade_MPa : 0;
            double maop = inputs.MaximumAllowableOperatingPressure_kPa != null ? (double)inputs.MaximumAllowableOperatingPressure_kPa : 0;
            DateTime pipeInstallDate = inputs.InstallationDate.Value;
            DateTime iliDate = inputs.ILIDate.Value;

            // following values checked for null in CanCalculate method, so it's safe to just obtain values.
            double? Pmin = inputs.Pmin_kPa;
            double? Pmax = inputs.Pmax_kPa;
            double? EquivalentPressureCycle = inputs.EquivalentPressureCycle;

            string anomalyType = inputs.AnomalyType;
            string iliCompany = inputs.ILICompany;

            // Average Ultimate Tensile Strength
            double UTS = AverageUltimateTensileStrength(pipeGrade);

            // Maximum Stress in Equivalent Stress Cycle (Mpa)
            double MaxStress_ESC;
            if (Pmax == null)
            {
                MaxStress_ESC = 483;
            }
            else
            {
                MaxStress_ESC = (double)Pmax * (double)od * Convert_Inch_To_MM / (2000 * (double)wt);
            }

            // Minimum Stress in Equivalent Stress Cycle (Mpa)
            double MinStress_ESC;
            if (Pmin == null)
            {
                MinStress_ESC = 0;
            }
            else
            {
                MinStress_ESC = (double)Pmin * (double)od * Convert_Inch_To_MM / (2000 * (double)wt);
            }

            // Sigma K (MPa)
            double SigmaK = MaxStress_ESC - MinStress_ESC;

            // Gouge Depth, measured (mm)
            double GougeDepthM = (double)wt * ((gougeDepth == null) || gougeDepth < 0.1 ? 0.1 : (double)gougeDepth);

            // Dent Width, standard deviation (mm)
            double DentW_sd;
            if (IsBaker(iliCompany))
            {
                DentW_sd = c_DentW_sd_BJ;
            }
            else
            {
                if (IsRosen(iliCompany))
                {
                    DentW_sd = c_DentW_sd_Rosen;
                }
                else
                {
                    DentW_sd = c_DentW_sd_TDW;
                }
            }

            // Dent Length, standard deviation (mm)
            double DentL_sd;
            if (IsBaker(iliCompany))
            {
                DentL_sd = c_DentL_sd_BJ;
            }
            else
            {
                if (IsRosen(iliCompany))
                {
                    DentL_sd = c_DentL_sd_Rosen;
                }
                else
                {
                    DentL_sd = c_DentL_sd_TDW;
                }
            }

            // Dent Depth, standard deviation, TDW (mm)
            double DentD_sd_TDW = c_DentD_sdP_TDW * Math.Floor((double)od) * Convert_Inch_To_MM;

            // Dent Depth, standard deviation fraction, Rosen
            double DentD_sdP_Rosen = StandardDeviation_Rosen((double)od);

            // Dent Depth, standard deviation, Rosen (mm)
            double DentD_sd_Rosen = DentD_sdP_Rosen * Math.Floor((double)od) * Convert_Inch_To_MM;

            // Dent Depth, measured (mm)
            double DentDepthM;
            if (dentDepth == null)
            {
                DentDepthM = 0.05 * Math.Floor((double)od) * Convert_Inch_To_MM;
            }
            else
            {
                DentDepthM = (double)od * (double)dentDepth * Convert_Inch_To_MM;
            }

            // Dent Depth, standard deviation (mm)
            double DentD_sd;
            if (IsBaker(iliCompany))
            {
                DentD_sd = c_DentD_sd_BJ;
            }
            else
            {
                if (IsRosen(iliCompany))
                {
                    DentD_sd = DentD_sd_Rosen;
                }
                else
                {
                    DentD_sd = DentD_sd_TDW;
                }
            }

            // Pressure Cycles since ILI
            double CyclesSinceILI;
            if (EquivalentPressureCycle == null)
            {
                CyclesSinceILI = 10;
            }
            else
            {
                CyclesSinceILI = (double)EquivalentPressureCycle * (DateTime.Now - pipeInstallDate).TotalDays / 365.25;
            }

            // create instance of failure flags...
            var allFlags = new ConcurrentBag<double[,]>();

            //  This Parallel.For() construct accumulates counts in the TFailureFlags class and periodically (but not every
            //  iteration) updates the flags instance from the threadflags instance and creates a new threadflags instance.
            Parallel.For(0, numberOfIterations,
                () => new double[FlagsArrayCount, 1],
                (j, loop, threadflags) =>
                {
                    // UTS, distributed (MPa)
                    double UTS_p = Normal.InvCDF(
                        UTS * c_UTS_m,
                        UTS * c_UTS_sd,
						SystemRandomSource.Default.NextDouble());

                    // Outside Diameter, distributed (mm)
                    double D_p = Normal.InvCDF(
                        (double)od * c_D_m,
                        (double)od * c_D_sd,
						SystemRandomSource.Default.NextDouble()) * Convert_Inch_To_MM;

                    // Wall thickness, distributed (mm)
                    double Wt_p = Normal.InvCDF(
                        (double)wt * c_Wt_m,
                        (double)wt * c_Wt_sd,
						SystemRandomSource.Default.NextDouble());

                    // Gouge Depth, distributed (mm)
                    double GougeDepth_p = 0;
                    if (anomalyType.ToLower().Contains("dent") && (anomalyType.ToLower().Contains("gouge") || anomalyType.ToLower().Contains("groove")))
                    {
                        GougeDepth_p = Math.Max(0, Normal.InvCDF(GougeDepthM * c_GougeD_m, (double)wt * c_GougeD_sd, SystemRandomSource.Default.NextDouble()));
                    }

                    // DentWidth, distributed (mm)
                    double DentWidth_p = Math.Max(0, Normal.InvCDF(dentLength * c_DentL_m, DentW_sd, SystemRandomSource.Default.NextDouble()));

                    // Dent Depth Operating, distributed (mm)
                    double DentDepth_p;
                    if ((dentDepth == null) || (dentDepth <= 0))
                    {
                        DentDepth_p = 0.01;
                    }
                    else
                    {
                        DentDepth_p = Math.Max(0.01, Normal.InvCDF(DentDepthM * c_DentD_m, DentD_sd, SystemRandomSource.Default.NextDouble()));
                    }

                    // DentLength, distributed(mm)
                    double DentLength_p = Math.Max(0, Normal.InvCDF(dentLength * c_DentL_m, DentL_sd, SystemRandomSource.Default.NextDouble()));

                    // Dent Radius, length (mm)
                    double DentRLength = (Math.Pow(DentLength_p, 2) + (4 * Math.Pow(DentDepth_p, 2))) / (8 * DentDepth_p);

                    // Dent Radius, width (mm)
                    double DentRWidth = (Math.Pow(DentWidth_p, 2) + (4 * Math.Pow(DentDepth_p, 2))) / (8 * DentDepth_p);

                    // Dent Radius (mm)
                    double DentRadius = Math.Min(DentRWidth, DentRLength);

                    // Dent Radius Coefficient
                    double DentRC;
                    if (DentRadius > Wt_p)
                    {
                        DentRC = 2;
                    }
                    else
                    {
                        DentRC = 1;
                    }

                    // Stress Enhancement Factor, dent
                    double Kd = 1 + DentRC * Math.Sqrt(Math.Pow(DentDepth_p, 1.5) * Wt_p / D_p);

                    // Stress Enhancement Factor, gouge
                    double Kg = 1 + 9 * (GougeDepth_p / Wt_p);

                    // Sigma A (MPa)
                    double SigmaA = SigmaK / (1 - Math.Pow((MaxStress_ESC + MinStress_ESC) / (2 * UTS_p), 2));

                    // Stress Cycles to Failure
                    double StressCycleToFailure = 5622 * Math.Pow(UTS_p / (SigmaA * Kd * Kg), 5.26);

                    // Leak flag
                    // For a purpose of this algorithm - treat everything as leaks
                    int leakFlag = (StressCycleToFailure < CyclesSinceILI) ? 1 : 0;

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
            double probabilityDentFailure = flags[LeakFlags_index, 0] * 1.0 / flags[Iterations_index, 0];
            outputs.FailureProbability = probabilityDentFailure;

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
        protected override IMechanicalDamageResidentCalculatorInputs LoadInputs(DataRow row)
        {
            if (row == null)
            {
                throw new ArgumentNullException(nameof(row));
            }

            IMechanicalDamageResidentCalculatorInputs inputs = new MechanicalDamageResidentCalculatorInputs();

            inputs.NumberOfIterations = (int)row[nameof(IMechanicalDamageResidentCalculatorInputs.NumberOfIterations)];
            inputs.NumberOfYears = (int)row[nameof(IMechanicalDamageResidentCalculatorInputs.NumberOfYears)];
            inputs.Id = Convert.ToInt32(row[nameof(IMechanicalDamageResidentCalculatorInputs.Id)].ToString());
            inputs.RangeId = Convert.ToInt32(row[nameof(IMechanicalDamageResidentCalculatorInputs.RangeId)].ToString());
            inputs.NominalWallThickness_mm = row[nameof(IMechanicalDamageResidentCalculatorInputs.NominalWallThickness_mm)].IfNullable<double>();
            inputs.OutsideDiameter_in = row[nameof(IMechanicalDamageResidentCalculatorInputs.OutsideDiameter_in)].IfNullable<double>();
            inputs.PipeGrade_MPa = row[nameof(IMechanicalDamageResidentCalculatorInputs.PipeGrade_MPa)].IfNullable<double>();
            inputs.Toughness_Joule = row[nameof(IMechanicalDamageResidentCalculatorInputs.Toughness_Joule)].IfNullable<double>();
            inputs.MaximumAllowableOperatingPressure_kPa = row[nameof(IMechanicalDamageResidentCalculatorInputs.MaximumAllowableOperatingPressure_kPa)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectLength_mm = row[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectLength_mm)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectWidth_mm = row[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectWidth_mm)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectDepthPrc = row[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)].IfNullable<double>();
            inputs.ILIMeasuredClusterDefectDepthPrcNPS = row[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectDepthPrcNPS)].IfNullable<double>();
            inputs.SurfaceIndicator = row[nameof(IMechanicalDamageResidentCalculatorInputs.SurfaceIndicator)].ToString();
            inputs.AnomalyType = row[nameof(IMechanicalDamageResidentCalculatorInputs.AnomalyType)].ToString();
            inputs.EquivalentPressureCycle = row[nameof(IMechanicalDamageResidentCalculatorInputs.EquivalentPressureCycle)].IfNullable<double>();
            inputs.Pmin_kPa = row[nameof(IMechanicalDamageResidentCalculatorInputs.Pmin_kPa)].IfNullable<double>();
            inputs.Pmax_kPa = row[nameof(IMechanicalDamageResidentCalculatorInputs.Pmax_kPa)].IfNullable<double>();

            inputs.InstallationDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(IMechanicalDamageResidentCalculatorInputs.InstallationDate)].ToString());
            inputs.ILIDate = SimulationCalculatorHelper.GetDateFromString(row[nameof(IMechanicalDamageResidentCalculatorInputs.ILIDate)].ToString());
            inputs.ILICompany = row[nameof(IMechanicalDamageResidentCalculatorInputs.ILICompany)].ToString();

            return inputs;
        }

        /// <summary>
        /// Creates the output table columns.
        /// </summary>
        /// <param name="inputsTable">The inputs table.</param>
        /// <param name="outputsTable">The outputs table.</param>
        /// <param name="outputs">The outputs.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputsTable" />, <paramref name="outputsTable" />, <paramref name="outputs"/> parameter is <b>null</b>.</exception>
        protected override void CreateOutputTableColumns(DataTable inputsTable, DataTable outputsTable, IMechanicalDamageResidentCalculatorOutputs outputs)
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
            outputsTable.Columns.Add(nameof(IMechanicalDamageResidentCalculatorOutputs.Elapsed), typeof(double));
            outputsTable.Columns.Add(nameof(IMechanicalDamageResidentCalculatorOutputs.Calculated), typeof(string));
            outputsTable.Columns.Add(nameof(IMechanicalDamageResidentCalculatorOutputs.ErrorMessage), typeof(string));

            outputsTable.Columns.Add(nameof(IMechanicalDamageResidentCalculatorOutputs.FailureProbability), typeof(double));
        }

        /// <summary>
        /// Saves the input and outputs in passed data row.
        /// </summary>
        /// <param name="inputs">The inputs row.</param>
        /// <param name="outputs">The outputs.</param>
        /// <param name="outputsRow">The row.</param>
        /// <exception cref="System.ArgumentNullException">Throw when <paramref name="inputs"/>, <paramref name="outputs"/>, <paramref name="outputsRow"/> parameter is <b>null</b>.</exception>
        protected override void SaveInputOutputs(IMechanicalDamageResidentCalculatorInputs inputs, IMechanicalDamageResidentCalculatorOutputs outputs, DataRow outputsRow)
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
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.NumberOfIterations)] = inputs.NumberOfIterations;
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.NumberOfYears)] = inputs.NumberOfYears;
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.Id)] = inputs.Id;
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.RangeId)] = inputs.RangeId;
            if (inputs.NominalWallThickness_mm != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.NominalWallThickness_mm)] = inputs.NominalWallThickness_mm;
            }
            if (inputs.OutsideDiameter_in != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.OutsideDiameter_in)] = inputs.OutsideDiameter_in;
            }
            if (inputs.PipeGrade_MPa != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.PipeGrade_MPa)] = inputs.PipeGrade_MPa;
            }
            if (inputs.Toughness_Joule != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.Toughness_Joule)] = inputs.Toughness_Joule;
            }
            if (inputs.MaximumAllowableOperatingPressure_kPa != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.MaximumAllowableOperatingPressure_kPa)] = inputs.MaximumAllowableOperatingPressure_kPa;
            }
            if (inputs.ILIMeasuredClusterDefectLength_mm != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectLength_mm)] = inputs.ILIMeasuredClusterDefectLength_mm;
            }
            if (inputs.ILIMeasuredClusterDefectWidth_mm != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectWidth_mm)] = inputs.ILIMeasuredClusterDefectWidth_mm;
            }
            if (inputs.ILIMeasuredClusterDefectDepthPrc != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectDepthPrc)] = inputs.ILIMeasuredClusterDefectDepthPrc;
            }
            if (inputs.ILIMeasuredClusterDefectDepthPrcNPS != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.ILIMeasuredClusterDefectDepthPrcNPS)] = inputs.ILIMeasuredClusterDefectDepthPrcNPS;
            }
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.SurfaceIndicator)] = inputs.SurfaceIndicator;
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.AnomalyType)] = inputs.AnomalyType;
            if (inputs.EquivalentPressureCycle != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.EquivalentPressureCycle)] = inputs.EquivalentPressureCycle;
            }
            if (inputs.Pmin_kPa != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.Pmin_kPa)] = inputs.Pmin_kPa;
            }
            if (inputs.Pmax_kPa != null)
            {
                outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.Pmax_kPa)] = inputs.Pmax_kPa;
            }
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.InstallationDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.InstallationDate);
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.ILIDate)] = SimulationCalculatorHelper.GetSringFromDate(inputs.ILIDate);
            outputsRow[nameof(IMechanicalDamageResidentCalculatorInputs.ILICompany)] = inputs.ILICompany;

            // output common data
            outputsRow[nameof(IMechanicalDamageResidentCalculatorOutputs.Elapsed)] = outputs.Elapsed;
            outputsRow[nameof(IMechanicalDamageResidentCalculatorOutputs.Calculated)] = outputs.Calculated ? "Y" : "N";
            outputsRow[nameof(IMechanicalDamageResidentCalculatorOutputs.ErrorMessage)] = outputs.ErrorMessage;

            outputsRow[nameof(IMechanicalDamageResidentCalculatorOutputs.FailureProbability)] = outputs.FailureProbability;
        }
        #endregion
    }
}
