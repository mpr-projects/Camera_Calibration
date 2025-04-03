This repository contains the code that I used in the video https://youtu.be/KGeRoScMs6Q. To use it create a settings file such as the one shown in the test directory. Depending on the target type that you choose you can/have to provide certain parameters in the settings file. Check out the file *target_functions.py*, there you can see which parameters are loaded.

For the Charuco pattern you can currently only use predefined dicts, not custom ones. I've only added a couple of them to the mapping, to add more check out *aruco_dict_mapping* in *target_functions.py*.

After you've copied your images into the folders specified in the settings file you can run the calibration with `python main.py path/to/settings/file`. To visualize the point distribution of your images run `python plot_point_coverage.py /path/to/settings/file`. To visualize the errors at the validation points run `python evaluate_errors.py path/to/pickle/output/file/of/calibration`.

The files also have some additional command line paramters that you can pass. Add `--help` to the commands to see them.

The camera matrix and distortion parameters are currently saved in a dictionary called *state*, which is saved in the pickle output file. I should probably add some code to easily extract it. For the time being just open the yaml file in Python and extract the state manually. There's one set of coefficients for each of the calibrations (each used one more picture than the one before).

Note, this code is not optimized and not tested thoroughly. So it's not super fast and it may contain bugs.
