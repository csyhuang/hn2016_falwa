
## Author - P Barpanda, Dec. 14, 2023

0) The sample data is in the dropbox link 
- https://www.dropbox.com/scl/fo/tx1vbwl584unzl5qjjrmv/h?rlkey=wxqu67zb5lp9xsnfx6zogfjuw&dl=0
Download the netcdf file and save it inside ./NorESM_data_sample/

## Run the python script using one of the two ways.

1) Execute shell script to read U, V, T variables together from the netcdf file and process the dataset.
Command --> nohup ./submit.sh > out.o 2> out.e

Or

2) Run the python script directly one by one for each variable, namely "U", "V", "T" as follows
Command --> python vertical_interpolation_hybrid_to_pressure_poisson_filled_below_topo.py "U"
