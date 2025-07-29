
MODEL_COLOURS = {
    "EC-Earth3": "red",
    "CESM2": "blue",
    "MPI-ESM1-2-LR": "gray",
    "CanESM5": "green",
    "ACCESS-ESM1-5": "orange",
    "Observations": "k--"
}
MODEL_START_YEAR = 1970
MODEL_END_YEAR = 2100
MODELS = [
    "EC-Earth3",
    "CESM2",
    "MPI-ESM1-2-LR",
    "CanESM5",
    "ACCESS-ESM1-5",
    "Observations",
]
MULTI_LIANG_RES_NAMES = [
    "T",
    "tau",
    "R",
    "error_T",
    "error_tau",
    "error_R",
]
SIG_LVL = 0.05
VARIABLES = [
    "ssie",     # September sea ice extent
    "wsie",     # March sea ice extent
    "wsiv",     # March sea ice volume
    "tas",      # 
    "oht_atl",  # Atlantic ocean heat transport
    "oht_pac",  # Pacific ocean heat transport
    "swfd",     # 
    "lwfd",
]
VAR_OBS_START_YEARS = {
    "ssie": 1979,
    "wsie": 1979, 
    "wsiv": 1979,
    "tas": 1970,
    "oht_atl": 1979,
    "oht_pac": 1979,
    "swfd": 2001,
    "lwfd": 2001,
}
VAR_LEGEND_ARGS = {
    "ssie": dict(loc='upper right',fontsize=24,shadow=True,frameon=False),
    "wsie": dict(loc='lower left',fontsize=24,shadow=True,frameon=False),
    "wsiv": dict(loc='upper right',fontsize=24,shadow=True,frameon=False),
    "tas": dict(loc='upper left',fontsize=24,shadow=True,frameon=False),
    "oht_atl": dict(loc='upper left',fontsize=23,shadow=True,frameon=False,ncol=2),
    "oht_pac": dict(loc='upper left',fontsize=24,shadow=True,frameon=False),
    "swfd": dict(loc='upper left',fontsize=23,shadow=True,frameon=False,ncol=2),
    "lwfd": dict(loc='upper left',fontsize=23,shadow=True,frameon=False,ncol=2),
}
VAR_YLABELS = {
    "ssie": 'September sea-ice extent (10$^6$ km$^2$)',
    "wsie": 'March sea-ice extent (10$^6$ km$^2$)', 
    "wsiv": 'March sea-ice volume (10$^3$ km$^3$)',
    "tas": 'Annual mean air temperature ($^\circ$C)',
    "oht_atl": 'Annual mean Atlantic OHT (TW)',
    "oht_pac": 'Annual mean Pacific OHT (TW)',
    "swfd": 'Annual mean net SWR (W m$^{-2}$)',
    "lwfd": 'Annual mean downward LWR (W m$^{-2}$)',
}
VAR_YLABELS_SHORT = {
    "ssie": "SSIE",
    "wsie": "WSIE",
    "tas": "$T_{2m}$",
    "wsiv": "WSIV",
    "oht_atl": "$OHT_{ATL}$",
    "oht_pac": "$OHT_{PAC}$",
}
VAR_YLIMITS = {
    "ssie": {"ymin": -0.5, "ymax": 12},
    "wsie": {"ymin": -1, "ymax": 21},
    "wsiv": {"ymin": -3, "ymax": 60},
    "tas": {"ymin": -18, "ymax": 5},
    "oht_atl": {"ymin": -10, "ymax": 270},
    "oht_pac": {"ymin": -5, "ymax": 70},
    "swfd": {"ymin": 25, "ymax": 65},
    "lwfd": {"ymin": 200, "ymax": 320},
}

