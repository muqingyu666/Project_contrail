#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#                       神兽保佑
#                      代码无BUG!

"""

    Function for filter anormal hcf data within lowermost 10%
    or highermost 90%.
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2022-10-26
    
    Including the following parts:

        1) Read in basic PCA & Cirrus data (include cirrus morphology and microphysics)  
        
        2) Filter anormal hcf data within lowermost 10% or highermost 90%

        
"""

import numpy as np

# ----------  importing dcmap from my util ----------#

# ---------- import done ----------#
# ---------------------------------#

############################################
########## Filt the error data #############
############################################


def filter_data_PC1_gap_lowermost_highermost_error(
    Cld_match_pc_gap_data,
):
    """
    Filter data within PC1 interval
    for 10% lowermost and 90% highermost

    Args:
        Cld_match_pc_gap_data (np.array): Cld data
        within PC1 interval shape in (PC_interval, 180, 360)
        PC_interval (int): PC1 interval numbers

    Returns:
        Cld_lowermost_error (np.array): Cld error data lowermost 10%
        Cld_highermost_error (np.array): Cld error data highermost 90%
        Cld_filtered (np.array): Cld filtered data
    """
    # make aux empty array
    Cld_lowermost_error = np.zeros(Cld_match_pc_gap_data.shape)
    Cld_highermost_error = np.zeros(Cld_match_pc_gap_data.shape)
    Cld_filtered = np.zeros(Cld_match_pc_gap_data.shape)

    # make aux cld data array
    Cld_data_match_pc_gap_aux = np.copy(
        Cld_match_pc_gap_data
    ).reshape(
        Cld_match_pc_gap_data.shape[0],
        Cld_match_pc_gap_data.shape[1]
        * Cld_match_pc_gap_data.shape[2],
    )

    PC_interval = Cld_match_pc_gap_data.shape[0]

    # loop over PC1 interval
    for PC_gap in range(PC_interval):
        Cld_lowermost_error[PC_gap, :, :] = np.array(
            np.where(
                (
                    np.nanpercentile(
                        Cld_data_match_pc_gap_aux[PC_gap, :], 10
                    )
                    <= Cld_match_pc_gap_data[PC_gap, :, :]
                ),
                -999,
                Cld_match_pc_gap_data[PC_gap, :, :].astype(
                    "float64"
                ),
            )
        )
        Cld_highermost_error[PC_gap, :, :] = np.array(
            np.where(
                (
                    Cld_match_pc_gap_data[PC_gap, :, :]
                    <= np.nanpercentile(
                        Cld_data_match_pc_gap_aux[:, PC_gap], 90
                    )
                ),
                -999,
                Cld_match_pc_gap_data[PC_gap, :, :].astype(
                    "float64"
                ),
            )
        )
        Cld_filtered[PC_gap, :, :] = np.array(
            np.where(
                (
                    np.nanpercentile(
                        Cld_data_match_pc_gap_aux[PC_gap, :], 10
                    )
                    <= Cld_match_pc_gap_data[PC_gap, :, :]
                )
                & (
                    Cld_match_pc_gap_data[PC_gap, :, :]
                    <= np.nanpercentile(
                        Cld_data_match_pc_gap_aux[PC_gap, :], 90
                    )
                ),
                Cld_match_pc_gap_data[PC_gap, :, :].astype(
                    "float64"
                ),
                -999,
            )
        )

    # small error data(within 10%) for cld
    Cld_lowermost_error[Cld_lowermost_error == -999] = np.nan

    # large error data(within 90%) for cld
    Cld_highermost_error[Cld_highermost_error == -999] = np.nan

    # filtered data for cld
    Cld_filtered[Cld_filtered == -999] = np.nan

    return Cld_lowermost_error, Cld_highermost_error, Cld_filtered


if __name__ == "__main__":
    print(
        "This is a function for filter anormal hcf data within lowermost 10% or highermost 90%"
    )
