import yaml, argparse
from pathlib import Path
from utilizes import EddyProc


def main():
    parser = argparse.ArgumentParser(description = "Do you have a complete yaml proclist or prefer param configuration?")
    parser.add_argument('d', type=str, help="yaml dir")
    parser.add_argument('--t', type=str, default = "p", help="transcript type, p for param configured, c for complete.")
    parser.add_argument('--o', type=str, default="./output", help="output folder")

    args = parser.parse_args()
    # yaml_dir = "half_complete_proc_list.yaml"
    # yaml_dir = "config.yaml"
    yaml_dir = args.d
    out_folder = Path(args.o)
    # print(out_folder)
    if not out_folder.is_absolute():
        if ".." in [p.as_posix() for p in out_folder.parents]:
            out_folder = Path.cwd().parent.joinpath(out_folder.name)
        else:
            out_folder = Path.cwd().joinpath(out_folder.name)
            # debug:
            # print("no ..")
            # print(list(out_folder.parents))
            pass
    else:
        # debug:
        # print("it's abs")
        pass

    parents = list(out_folder.parents)
    parents.append(out_folder)
    parents.reverse()
    for par in parents:
        if not par.exists(): 
            par.mkdir()
        else:
            print(f"{par} exists")

    eddy = EddyProc()
    print("Starting...")
    with open(yaml_dir, 'r') as stream:
        try:
            user_config = yaml.safe_load(stream)
            if args.t == 'c':
                transcrpit_yaml_list(user_config, eddy, out_folder)
            elif args.t == 'p':
                parse_yaml_params(user_config, eddy, out_folder)
            else:
                raise Exception("please specify the correct transcript type!")
        except yaml.YAMLError as exc:
            print(exc)
    
    print("Finished successfully.")

def parse_yaml_params(inyaml, eddy, out_folder, outfile = "proc_list.txt", outyaml = "variables.yaml"):

    invars = inyaml["vars"]
    if "pos" in invars.keys():
        # do something...
        channels = invars["pos"]
        pass
    elif "pos_begin" in invars.keys():
        channels = [invars["pos_begin"] + idx for idx in range(len(invars["labels"]))]
    else:
        raise Exception("No channel info!")
    # dCalc = inyaml["outdir"]["dCalc"]
    # dSpec = inyaml["outdir"]["dSpec"]
    results = "" # change to "".join(list)
    cur_vars = {}
    cur_vars["vars_chns"] = dict(zip(invars["labels"], channels))
    cur_vars["vars_chns"].update(inyaml["outdir"])
    cur_vars["in_vars"] = invars["labels"]
    ######## START PROC LIST
    results += eddy.location_output_files(dCalc = inyaml["outdir"]["dCalc"], dSpec= inyaml["outdir"]["dSpec"])
    results += eddy.comments(cmt1 = "Extract the data")
    for idx, pos in enumerate(channels):
        results += eddy.extract(chn = pos, label = invars["labels"][idx])
    results += eddy.comments(cmt1 = "Convert cell (c) and block (b) RH's to absolute densities")
    for label in ["cH2O", "c_e"]:
        results += eddy.extract(chn = cur_vars["vars_chns"]["cRH"], label = label)
        cur_vars["vars_chns"][label] = cur_vars["vars_chns"]["cRH"]
        cur_vars["in_vars"].append(label)
    for label in ["bH2O", "b_e"]:
        results += eddy.extract(chn = cur_vars["vars_chns"]["bRH"], label = label)
        cur_vars["vars_chns"][label] = cur_vars["vars_chns"]["bRH"]
        cur_vars["in_vars"].append(label)
    params = inyaml["humidity_unit_conv"] # intro callback function instead!!!.
    for param in params:
        results += eddy.gas_conversion_time_series(**param)
    results += eddy.comments(cmt1 = "Clean data")
    results += eddy.despike(sig = "Ts", replaceSpikes = "x")
    cur_vars["new_vars"]= ["x"]
    results += eddy.despike_vickers(sig = "Ts", count_ts_vickers = "count_Ts_Vickers")
    cur_vars["new_vars"].append("count_Ts_Vickers")
    results += eddy.comments(cmt1 = "Calculate some means")
    for var in cur_vars["in_vars"]:
        if var in ["Ux", "Uy", "Uz"]:
            new_mean_var = "mean_" + var + "_pre_rot"
            results += eddy.one_chn_statistics(sig = var, SLmean = new_mean_var)
            cur_vars["new_vars"].append(new_mean_var)
        elif var == "Ts":
            new_mean_var = "mean_" + var + "_virt"
            results += eddy.one_chn_statistics(sig = var, SLmean = new_mean_var)
            cur_vars["new_vars"].append(new_mean_var)
        elif var not in ["c_e", "b_e"]:
            new_mean_var = "mean_" + var
            results += eddy.one_chn_statistics(sig = var, SLmean = new_mean_var)
            cur_vars["new_vars"].append(new_mean_var)
    results += eddy.virtual_temperature_raw(sig = "Ts", sigH2O = "bH2O", PKpa = "mean_cP")
    for var in ["Ts", "c_e", "b_e"]:
            new_mean_var = "mean_" + var
            results += eddy.one_chn_statistics(sig = var, SLmean = new_mean_var)
            cur_vars["new_vars"].append(new_mean_var)
    results += eddy.comments(cmt1 = "Plot some of the means")
    params = inyaml["plot_mean"]
    for param in params:
        results += eddy.plot_value(**param)
    results += eddy.comments(
        cmt1 = "Calculate the wind direction before rotations",
        cmt2 = "Wind speed is calculated as part of the coordinate rotation",
        cmt3 = "Save Ux, Uy, and Uz Components before rotation"
    )
    cur_vars["new_vars"].append("Wind_Dir")
    cur_vars["new_vars"].append("Wind_Dir_Std")
    results += eddy.wind_direction(sigU="Ux", sigV="Uy", orient="0", SLWindDir = "Wind_Dir", SLWindDirDiv = "Wind_Dir_Std")
    results += eddy.comments("X corrs")

    params = inyaml["cross_corr"]
    for param in params:
        cur_vars["new_vars"].append(param["SLPeakTime"])
        results += eddy.cross_correlate(**param)
        results += eddy.plot_cross_correlation()
    
    results += eddy.comments("Remove Lags")

    params = inyaml["remove_lag"]
    for param in params:
        results += eddy.remove_lag(**param)

    results += eddy.comments("Coordinate Rotation")

    results += eddy.rotation_coefficients(sigU="Ux", sigV="Uy", sigW="Uz")

    results += eddy.rotation(sigU="Ux", sigV="Uy", sigW="Uz", do1Rot="x", do2Rot="x")

    results += eddy.one_chn_statistics(sig="Ux", SLmean="mean_Wind_Spd")
    cur_vars["new_vars"].append("mean_Wind_Spd")

    results += eddy.comments("Plot Spectra and Cospectra")

    for sig in ["Ts", "bH2O", "cH2O", "vCO2"]:
        results += eddy.spectra(sig=sig)
        results += eddy.cospectra(sig1="Uz", sig2=sig)
        results += eddy.plot_spectral()

    results += eddy.comments("CO2 from ppm to umol per m3")

    params = inyaml["co2_umol2density"]
    for param in params:
        results += eddy.gas_conversion_time_series(**param)

    results += eddy.comments("Calculate fluxes")

    results += eddy.sensible_heat_flux_coefficient(
        SLable="rhocp", vaporPres="mean_b_e", temp="mean_Ts", pres="mean_cP"
        )
    cur_vars["new_vars"].append("rhocp")

    results += eddy.latent_heat_of_evaporation(
        SLable="lambda", temp = "mean_Ts", pres = "mean_cP"
        )
    cur_vars["new_vars"].append("lambda")

    results += eddy.friction_velocity(
        sigU="Ux", sigV="Uy", sigW="Uz", SLuwvw = "friction_velocity"
        )
    cur_vars["new_vars"].append("friction_velocity")

    results += eddy.plot_value(LVal = "mean_Wind_Spd", RVal = "friction_velocity")

    params = inyaml["cal_flux_coef"]
    for param in params:
        results += eddy.two_chn_statistics(**param)
        cur_vars["new_vars"].append(param["SLFlux"])

    params = inyaml["plot_flux"]
    for param in params:
        results += eddy.plot_value(**param)

    results += eddy.comments("Frequency response correction")

    params = inyaml["tube_atten"]
    for param in params:
        results += eddy.tube_attenuation(**param)
        cur_vars["new_vars"].append(param["SLable"])

    param = inyaml["set_values"]
    cur_vars["set_values"] = param
    results += eddy.set_value(valDict = param)
    # for p_item in param.items():
    #     cur_vars["set_values"][p_item[0]] = p_item[1]

    results += eddy.mathematical_operation(
        SLable = "irga_path", measuredVarA = "mean_Wind_Spd",
        operation = "*", measuredVarB = "0.1"
    )
    cur_vars["new_vars"].append("irga_path")

    results += eddy.user_defined(
        SLable = "irga_path2", 
        equation = "Max(irga_path,0.005)",
        var = "irga_path"
    )
    cur_vars["new_vars"].append("irga_path2")

    param = inyaml["monin_obhukov"]
    results += eddy.stability_monin_obhukov(**param)
    cur_vars["new_vars"].append(param["SLable"])
    results += eddy.plot_value(LVal = "MO_stability")

    params = inyaml["freq_resp"]
    for param in params:
        results += eddy.frequency_response(**param)
        cur_vars["new_vars"].append(param["SLable"])
    
    params = inyaml["plot_freq_resp"]
    for param in params:
        results += eddy.plot_value(**param)

    params = inyaml["math_freq_oper"]
    for param in params:
        results += eddy.mathematical_operation(**param)
        cur_vars["new_vars"].append(param["SLable"])

    params = inyaml["plot_math_freq_oper"]
    for param in params:
        results += eddy.plot_value(**param)

    # Write lines and variables out

    with open(out_folder.joinpath(outfile), "w") as f:
        f.write(results[0: -1])

    temp = {}
    temp["attr"] = {
        "vars": {
            "in_vars": cur_vars["in_vars"],
            "new_vars": cur_vars["new_vars"]
        },
        "user_define": {
            "vars_chns": cur_vars["vars_chns"],
            "set_values": cur_vars["set_values"]
        }
    }
    cur_vars = temp

    with open(out_folder.joinpath(outyaml), 'w') as f:
        yaml.dump(cur_vars, f, default_flow_style = False)

    # debug:
    # print(list(cur_vars.keys()))
    print("done")

def transcrpit_yaml_list(yamlfile, eddy, out_folder, outfile = "transcripted_list.txt"):
    results = ""
    blocks = yamlfile["eddy_proc_list"]
    # print(blocks)
    # vars_dict = {}
    # vars_dict.update()
    for block in blocks:
        func_name = block[0]
        func_vars = block[1]
        # print(func_vars)
        # print(func_name)
        # print(func_vars)
        func = getattr(eddy, func_name)
        for func_var in func_vars:
            # print(func_var)
            results += func(**func_var)
    #     exit(0)
    # print(results)
    with open(out_folder.joinpath(outfile), "w") as f:
        f.write(results[0: -1])
    print("done")

if __name__ == "__main__":
    main()