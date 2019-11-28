
class EddyProc(object):
    super(object)
    def __init__(self):
        # Constants and default parameters.
        # self.offset1 = 0
        # self.gain1 = 1
        # self.curvature1 = 0
        pass

    # Choices for processing list
    def location_output_files(self, dCalc = "", dSpec = "", dWave = "", dCroCor = "", 
                            dDis = "", dQuad = "", dRef = ""):
        return (
            f"Location Output Files\n\t"
                f"Output File Calculations = {dCalc}\n\t"
                f"Output File Spectral = {dSpec}\n\t"
                f"Output File Wavelet = {dWave}\n\t"
                f"Output File Cross Correlation = {dCroCor}\n\t"
                f"Output File Distribution = {dDis}\n\t"
                f"Output File Quadrant = {dQuad}\n\t"
                f"Output File Reference = {dRef}\n"
        )

    def comments(self, cmt1 = "", cmt2 = "", cmt3 = ""):
        return (
            f"Comments\n\t"
                f"Comment = {cmt1}\n\t"
                f"Comment = {cmt2}\n\t"
                f"Comment = {cmt3}\n"
        )

    def extract(self, sTime = "", eTime = "", chn = None, label = ""):
        return (
            f"Extract\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Channel = {chn}\n\t"
                f"Label for Signal = {label}\n"
        )

    def gas_conversion_time_series(self, sTime = "", eTime = "", sig = "", 
                                    conFrom = "", conTo = "",
                                    offset1 = "0", gain1 = "1", curvature1 = "0", sigTC = "",
                                    valTC = "", sigPkPa = "", valPkPa = "", sigH2O = "", 
                                    valH2O = "", unitsH2O = "", moleWeight = "18",
                                    offset2 = "0", gain2 = "1", curvature2 = "0"):
        return(
            f"Gas conversion time series\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal = {sig}\n\t"
                f"Convert from = {conFrom}\n\t"
                f"Convert to = {conTo}\n\t"
                f"1st Offset = {offset1}\n\t"
                f"1st Gain = {gain1}\n\t"
                f"1st Curvature = {curvature1}\n\t"
                f"Signal T, C = {sigTC}\n\t"
                f"Value T, C = {valTC}\n\t"
                f"Signal P, kPa = {sigPkPa}\n\t"
                f"Value P, kPa = {valPkPa}\n\t"
                f"Signal H2O = {sigH2O}\n\t"
                f"Value H2O = {valH2O}\n\t"
                f"Units H2O = {unitsH2O}\n\t"
                f"Molecular Weight = {moleWeight}\n\t"
                f"2nd Offset = {offset2}\n\t"
                f"2nd Gain = {gain2}\n\t"
                f"2nd Curvature = {curvature2}\n"     
        )

    def despike(self, sTime = "", eTime = "", sig = "", std = "5", spikeWid = "4", consistency = "30",
                replaceSpikes = "", SLScount = "", outlierStd = "9"):
        return(
            f"Despike\n\t"
                f"From Time = {sTime}\n\t" 
                f"To Time = {eTime}\n\t" 
                f"Signal = {sig}\n\t"
                f"Standard Deviations = {std}\n\t"
                f"Spike width = {spikeWid}\n\t"
                f"Spike % consistency = {consistency}\n\t"
                f"Replace spikes = {replaceSpikes}\n\t"
                f"Storage Label spike count = {SLScount}\n\t"
                f"Outlier Standard Deviations = {outlierStd}\n"      
        )

    def despike_vickers(self, sTime = "", eTime = "", sig = "", winSize = "5", winStep = "10", 
                        iniStd = "3.5", increStd = "0.2", maxSpikeWid = "3", maxNum = "10",
                        count_ts_vickers = "", SLDSpikes = ""):
        return(
            f"Despike - Vickers\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {sTime}\n\t" 
                f"Signal = {sig}\n\t"
                f"Window size, minutes = {winSize}\n\t"
                f"Window step, points =  {winStep}\n\t"
                f"Initial standard deviation = {iniStd}\n\t"
                f"Standard deviation increment, % = {increStd}\n\t"
                f"Max spike width = {maxSpikeWid}\n\t"
                f"Max number of passes = {maxNum}\n\t"
                f"Storage Label spike count = {count_ts_vickers}\n\t"
                f"Storage Label data % spikes = {SLDSpikes}\n"
        )

    def one_chn_statistics(self, sTime = "", eTime = "", sig = "", SLmean = "", SLStd = "", 
                            SLSkew = "", SLKurt = "", SLMax = "", SLMin = "", 
                            SLVar = "", SLTurInten = "", ATIDenomi = ""):
        return(
            f"1 chn statistics\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal = {sig}\n\t"
                f"Storage Label Mean = {SLmean}\n\t"
                f"Storage Label Std Dev = {SLStd}\n\t"
                f"Storage Label Skewness = {SLSkew}\n\t"
                f"Storage Label Kurtosis = {SLKurt}\n\t"
                f"Storage Label Maximum = {SLMax}\n\t"
                f"Storage Label Minimum = {SLMin}\n\t" 
                f"Storage Label Variance = {SLVar}\n\t"
                f"Storage Label Turbulent Intensity = {SLTurInten}\n\t"
                f"Alt Turbulent Intensity Denominator = {ATIDenomi}\n"        
        )

    def virtual_temperature_raw(self, sTime = "", eTime = "", sig = "", sigH2O = "", PKpa = "",
                                wVapourUnit = "Absolute density g/m3", tempConver = "Calculate true from virtual-sonic"):
        return(
            f"Virtual Temperature Raw\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal T(C) = {sig}\n\t"
                f"Signal H2O = {sigH2O}\n\t"
                f"Pressure, kPa = {PKpa}\n\t"
                f"Water vapour units = {wVapourUnit}\n\t"
                f"Temperature conversion = {tempConver}\n"        
        )

    def plot_value(self, sTime = "", eTime = "", LVal = "", RVal = "", LMin = "", LMax = "",
                    RMin = "", RMax = "", match = ""):
        return(
            f"Plot Value\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Left Axis Value = {LVal}\n\t"
                f"Right Axis Value = {RVal}\n\t"
                f"Left Axis Minimum = {LMin}\n\t"
                f"Left Axis Maximum = {LMax}\n\t"
                f"Right Axis Minimum = {RMin}\n\t"
                f"Right Axis Maximum = {RMax}\n\t"
                f"Match Left/Right Axes = {match}\n"
        )

    def wind_direction(self, sTime = "", eTime = "", sigU = "", sigV = "", orient = "",
                        windDirCompo = "U+N_V+W", windDirOut = "N_0_deg-E_90_deg",
                        SLWindDir = "", SLWindDirDiv = ""):
        return(
            f"Wind direction\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal (u) = {sigU}\n\t"
                f"Signal (v) = {sigV}\n\t"
                f"Orientation = {orient}\n\t"
                f"Wind Direction Components = {windDirCompo}\n\t"
                f"Wind Direction Output = {windDirOut}\n\t"
                f"Storage Label Wind Direction = {SLWindDir}\n\t"
                f"Storage Label Wind Dir Std Dev = {SLWindDirDiv}\n"
        )

    def cross_correlate(self, sTime = "", eTime = "", sig = "", sigLag = "", corrType = "Covariance", 
                        corrCurveOut = "", SLPeakTime = "", SLPeakVal = ""):
        return(
            f"Cross Correlate\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal = {sig}\n\t"
                f"Signal which lags = {sigLag}\n\t"
                f"Correlation type = {corrType}\n\t"
                f"Output Correlation curve = {corrCurveOut}\n\t"
                f"Storage Label Peak Time = {SLPeakTime}\n\t"
                f"Storage Label Peak Value = {SLPeakVal}\n"
        )

    def plot_cross_correlation(self, sTime = "", eTime = ""):
        return(
            f"Plot cross correlation\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n"
        )

    def remove_lag(self, sTime = "", eTime = "", sig = "", minLagS = "", lagS = "", maxLagS = "",
                    belowMinDefault = "", belowMaxDefault = ""):
        return(
            f"Remove Lag\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal = {sig}\n\t"
                f"Min Lag (sec) =  {minLagS}\n\t"
                f"Lag (sec) =  {lagS}\n\t"
                f"Max Lag (sec) = {maxLagS}\n\t"
                f"Below Min default (sec) = {belowMinDefault}\n\t"
                f"Above Max default (sec) = {belowMaxDefault}\n"
        )

    def rotation_coefficients(self, sTime = "", eTime = "", sigU = "", sigV = "", sigW = "", 
                                SLAlpha = "alpha", SLBeta = "beta", SLGamma = "gamma",
                                optMeanU = "", optMeanV = "", optMeanW = ""):
        return(
            f"Rotation coefficients\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal (u) = {sigU}\n\t"
                f"Signal (v) = {sigV}\n\t"
                f"Signal (w) = {sigW}\n\t"
                f"Storage Label Alpha = {SLAlpha}\n\t"
                f"Storage Label Beta  = {SLBeta}\n\t"
                f"Storage Label Gamma = {SLGamma}\n\t"
                f"Optional mean u = {optMeanU}\n\t"
                f"Optional mean v = {optMeanV}\n\t"
                f"Optional mean w = {optMeanW}\n"
        )

    def rotation(self, sTime = "", eTime = "", sigU = "", sigV = "", sigW = "",
                alpha = "alpha", beta = "beta", gamma = "gamma",
                do1Rot = "", do2Rot = "", do3Rot = ""):
        return(
            f"Rotation\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal (u) = {sigU}\n\t"
                f"Signal (v) = {sigV}\n\t"
                f"Signal (w) = {sigW}\n\t"
                f"Alpha = {alpha}\n\t"
                f"Beta = {beta}\n\t"
                f"Gamma = {gamma}\n\t"
                f"Do 1st Rot = {do1Rot}\n\t"
                f"Do 2nd Rot = {do2Rot}\n\t"
                f"Do 3rd Rot = {do3Rot}\n"
        )

    def spectra(self, sTime = "", eTime = "", sig = "", wind = "Hamming", kaiser_bell = "", 
                specOut = "x", sortOut = "x", detrendData = "Low"):
        return(
            f"Spectra\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal = {sig}\n\t"
                f"Window = {wind}\n\t"
                f"Kaiser/Bell coef = {kaiser_bell}\n\t"
                f"Output Spectra = {specOut}\n\t"
                f"Sort Output = {sortOut}\n\t"
                f"Detrend data = {detrendData}\n"
        )

    def cospectra(self, sTime = "", eTime = "", sig1 = "", sig2 = "", wind = "Hamming", kaiser_bell = "", 
                cospecOut = "x", cohereOut = "", phaseOut = "", sortOut = "x", 
                detrendSig1 = "Low", detrendSig2 = "Low"):
        return(
            f"Cospectra\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Signal 1 = {sig1}\n\t"
                f"Signal 2 = {sig2}\n\t"
                f"Window = {wind}\n\t"
                f"Kaiser/Bell = {kaiser_bell}\n\t"
                f"Output Cospectra = {cospecOut}\n\t"
                f"Output Coherence = {cohereOut}\n\t"
                f"Output Phase = {phaseOut}\n\t"
                f"Sort Output = {sortOut}\n\t"
                f"Detrend data, Signal 1 = {detrendSig1}\n\t"
                f"Detrend data, Signal 2 = {detrendSig2}\n" 
        )

    def plot_spectral(self, sTime = "", eTime = "", LAxSpec = "Spectra", RAxCospec = "Cospectra", 
                    LAxLog = "x", RAxLog = "x", LAxMin = "", LAxMax = "", RAxMin = "", RAxMax = "", matchAxs = ""):
        return(
            f"Plot spectral\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Left Axis Spectra = {LAxSpec}\n\t"
                f"Right Axis Spectra = {RAxCospec}\n\t"
                f"Left Axis Logarithmic = {LAxLog}\n\t"
                f"Right Axis Logarithmic = {RAxLog}\n\t"
                f"Left Axis Minimum = {LAxMin}\n\t"
                f"Left Axis Maximum = {LAxMax}\n\t"
                f"Right Axis Minimum = {RAxMin}\n\t"
                f"Right Axis Maximum = {RAxMax}\n\t"
                f"Match Left/Right Axes = {matchAxs}\n"   
        )

    def sensible_heat_flux_coefficient(self, sTime = "", eTime = "", SLable = "", appTo = "", appBy = "",
                                        vaporPres = "", minQC = "", maxQC = "", temp = "",
                                        pres = "", altrhoCp = ""):
        return(
            f"Sensible heat flux coefficient\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {sTime}\n\t"
                f"Storage Label = {SLable}\n\t"
                f"Apply to = {appTo}\n\t"
                f"Apply by = {appBy}\n\t"
                f"Vapour pressure (KPa) = {vaporPres}\n\t"
                f"Min or QC = {minQC}\n\t"
                f"Max or QC = {maxQC}\n\t"
                f"Temperature (C) = {temp}\n\t"
                f"Min or QC = {minQC}\n\t"
                f"Max or QC = {maxQC}\n\t"
                f"Pressure (KPa) = {pres}\n\t"
                f"Min or QC = {minQC}\n\t"
                f"Max or QC = {maxQC}\n\t"
                f"Alternate rhoCp = {altrhoCp}\n" 
        )

    def latent_heat_of_evaporation(self, sTime = "", eTime = "", SLable = "", appTo = "", appBy = "",
                                    temp = "", minQC = "", maxQC = "", pres = "", LECoef = ""):
        return(
            f"Latent heat of evaporation\n\t"
                f"From Time = {sTime}\n\t" 
                f"To Time = {eTime}\n\t" 
                f"Storage Label = {SLable}\n\t"
                f"Apply to = {appTo}\n\t" 
                f"Apply by = {appBy}\n\t" 
                f"Temperature (C) = {temp}\n\t"
                f"Min or QC = {minQC}\n\t" 
                f"Max or QC = {maxQC}\n\t" 
                f"Pressure (KPa) = {pres}\n\t"
                f"Min or QC = {minQC}\n\t" 
                f"Max or QC = {maxQC}\n\t" 
                f"LE flux coef, L = {LECoef}\n" 
        )

    def friction_velocity(self, sTime = "", eTime = "", sigU = "", sigV = "", sigW = "",
                        SLuw = "", SLuwvw = ""):
        return(
            f"Friction Velocity\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t" 
                f"Signal (u) = {sigU}\n\t"
                f"Signal (v) = {sigV}\n\t"
                f"Signal (w) = {sigW}\n\t"
                f"Storage Label U* (uw) = {SLuw}\n\t"
                f"Storage Label U* (uw vw) = {SLuwvw}\n"
        )

    def two_chn_statistics(self, sTime = "", eTime = "", sig1 = "", sig2 = "", SLCov = "", SLCor = "",
                            SLFlux = "", fluxCoef = ""):
        return(
            f"2 chn statistics\n\t"
                f"From Time = {sTime}\n\t" 
                f"To Time = {eTime}\n\t" 
                f"Signal = {sig1}\n\t"
                f"Signal = {sig2}\n\t"
                f"Storage Label Covariance = {SLCov}\n\t"
                f"Storage Label Correlation = {SLCor}\n\t" 
                f"Storage Label Flux = {SLFlux}\n\t"
                f"Flux coefficient = {fluxCoef}\n"
        )

    def tube_attenuation(self, sTime = "", eTime = "", SLable = "", appTo = "", appBy = "",
                        gasSpecies = "", tubePres = "", minQC = "", maxQC = "", flowRate = "",
                        tubeLen = "", tubeID = "", uDefLambdaCoef = ""):
        return(
            f"Tube attenuation\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Storage Label = {SLable}\n\t"
                f"Apply to = {appTo}\n\t"
                f"Apply by = {appBy}\n\t"
                f"Gas species = {gasSpecies}\n\t"
                f"Tube pressure (KPa) = {tubePres}\n\t"
                f"Min or QC = {minQC}\n\t"
                f"Max or QC = {maxQC}\n\t"
                f"Flow rate (LPM) = {flowRate}\n\t"
                f"Tube length (m) = {tubeLen}\n\t"
                f"Tube ID (m) = {tubeID}\n\t"
                f"User defined Lambda coefficient = {uDefLambdaCoef}\n"
        )

    def set_value(self, valDict = {}, sTime = "", eTime = ""):
        items = valDict.items()
        varNum = len(items)
        retLines = (
            f"Set Values\n\t"
                f"From Time = {sTime}\n\t" 
                f"To Time = {sTime}\n\t"
                f"Number of Variables = {varNum}\n\t"
        )
        # print(type(retLines))
        for item in items:
            retLines += (
                f"Storage Label = {item[0]}\n\t" 
                f"Assignment value = {item[1]}\n\t"
            )
        ## \t should be cut out from the last line.
        # retLines = retLines[0: -1]
        return retLines[0: -1]

    def mathematical_operation(self, sTime = "", eTime = "",  SLable = "", appTo = "", appBy = "",
                                measuredVarA = "", operation = "", measuredVarB = ""):
        return(
            f"Mathematical operation\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Storage Label = {SLable}\n\t"
                f"Apply to = {appTo}\n\t"
                f"Apply by = {appBy}\n\t"
                f"Measured variable A =  {measuredVarA}\n\t"
                f"Operation  = {operation}\n\t"
                f"Measured variable B =  {measuredVarB}\n"
        )

    def user_defined(self, sTime = "", eTime = "",  SLable = "", appTo = "", appBy = "",
                    equation = "", var = ""):
        return(
            f"User defined\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t" 
                f"Storage Label = {SLable}\n\t"
                f"Apply to = {appTo}\n\t"
                f"Apply by = {appBy}\n\t" 
                f"Equation = {equation}\n\t"
                f"Variable = {var}\n\t"
        )

    def stability_monin_obhukov(self, sTime = "", eTime = "",  SLable = "", appTo = "", appBy = "",
                                measHeight = "", zeroPlaneDisp = "", virtualTemp = "",
                                minQC = "", maxQC = "", HFlux = "", HFluxCoef = "", fVelo = ""):
        return(
            f"Stability - Monin Obhukov\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Storage Label = {SLable}\n\t"
                f"Apply to = {appTo}\n\t"
                f"Apply by = {appBy}\n\t" 
                f"Measurement height (m) =  {measHeight}\n\t"
                f"Zero plane displacement (m) = {zeroPlaneDisp}\n\t"
                f"Virtual Temperature (C) = {virtualTemp}\n\t"
                f"Min or QC = {minQC}\n\t" 
                f"Max or QC = {maxQC}\n\t"
                f"H flux (W/m2) = {HFlux}\n\t"
                f"Min or QC = {minQC}\n\t"
                f"Max or QC = {maxQC}\n\t"
                f"H flux coef, RhoCp = {HFluxCoef}\n\t"
                f"Min or QC = {minQC}\n\t"
                f"Max or QC = {maxQC}\n\t"
                f"Scaling velocity (m/s) =  {fVelo}\n\t"
                f"Min or QC = {minQC}\n\t"
                f"Max or QC =  {maxQC}\n"       
        )

    def frequency_response(self, sTime = "", eTime = "",  SLable = "", appTo = "", appBy = "",
                        corrType = "", measHeight = "", zeroPlaneDisp = "", boundLyrHeight = "",
                        stability = "", windSpeed = "", 
                        sen1FlowVelo = "", sen1SampFreq = "",
                        sen1LPassFiltType = "", sen1LPassFiltTimeConstant = "",
                        sen1HPassFiltType = "", sen1HPassFiltTimeConstant = "",
                        sen1PathLen = "", sen1TimeConstant = "", sen1TubeAttenuCoef = "",
                        sen2FlowVelo = "", sen2SampFreq = "",
                        sen2LPassFiltType = "", sen2LPassFiltTimeConstant = "",
                        sen2HPassFiltType = "", sen2HPassFiltTimeConstant = "",
                        sen2PathLen = "", sen2TimeConstant = "", sen2TubeAttenuCoef = "",
                        pathSepera = "", getSpecDataType = "", getRespFuncFrom = "", 
                        refTag = "", refRespCond = "", sen1SubSamp = "", sen2SubSamp = "",
                        appVeloDistAdjust = "", useCalcDist = "", veloDistStd = "", stabilityDistStd = ""):
        return(
            f"Frequency response\n\t"
                f"From Time = {sTime}\n\t"
                f"To Time = {eTime}\n\t"
                f"Storage Label = {SLable}\n\t"
                f"Apply to = {appTo}\n\t" 
                f"Apply by = {appBy}\n\t"
                f"Correction type = {corrType}\n\t"
                f"Measurement height (m) = {measHeight}\n\t"
                f"Zero plane displacement (m) = {zeroPlaneDisp}\n\t"
                f"Boundary layer height (m) = {boundLyrHeight}\n\t"
                f"Stability Z/L = {stability}\n\t"
                f"Wind speed (m/s) = {windSpeed}\n\t"
                f"Sensor 1 Flow velocity (m/s) = {sen1FlowVelo}\n\t"
                f"Sensor 1 Sampling frequency (Hz) = {sen1SampFreq}\n\t"
                f"Sensor 1 Low pass filter type = {sen1LPassFiltType}\n\t"
                f"Sensor 1 Low pass filter time constant = {sen1LPassFiltTimeConstant}\n\t" 
                f"Sensor 1 High pass filter type = {sen1HPassFiltType}\n\t"
                f"Sensor 1 High pass filter time constant = {sen1HPassFiltTimeConstant}\n\t"
                f"Sensor 1 Path length (m) = {sen1PathLen}\n\t"
                f"Sensor 1 Time constant (s) = {sen1TimeConstant}\n\t"
                f"Sensor 1 Tube attenuation coef = {sen1TubeAttenuCoef}\n\t" 
                f"Sensor 2 Flow velocity (m/s) = {sen2FlowVelo}\n\t"
                f"Sensor 2 Sampling frequency (Hz) = {sen2SampFreq}\n\t"
                f"Sensor 2 Low pass filter type = {sen2LPassFiltType}\n\t"
                f"Sensor 2 Low pass filter time constant = {sen2LPassFiltTimeConstant}\n\t"
                f"Sensor 2 High pass filter type = {sen2HPassFiltType}\n\t"
                f"Sensor 2 High pass filter time constant = {sen2HPassFiltTimeConstant}\n\t"
                f"Sensor 2 Path length (m) = {sen2PathLen}\n\t"
                f"Sensor 2 Time constant (s) = {sen2TimeConstant}\n\t"
                f"Sensor 2 Tube attenuation coef = {sen2TubeAttenuCoef}\n\t"
                f"Path separation (m) = {pathSepera}\n\t"
                f"Get spectral data type = {getSpecDataType}\n\t"
                f"Get response function from = {getRespFuncFrom}\n\t"
                f"Reference Tag = {refTag}\n\t"
                f"Reference response condition = {refRespCond}\n\t" 
                f"Sensor 1 subsampled = {sen1SubSamp}\n\t"
                f"Sensor 2 subsampled = {sen2SubSamp}\n\t"
                f"Apply velocity distribution adjustment = {appVeloDistAdjust}\n\t" 
                f"Use calculated distribution = {useCalcDist}\n\t" 
                f"Velocity distribution std dev= {veloDistStd}\n\t" 
                f"Stability distribution std dev= {stabilityDistStd}\n" 
        )