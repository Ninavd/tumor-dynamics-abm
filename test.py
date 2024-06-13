import mesa_reader as mr

# make a MesaData object from a history file
h = mr.MesaData('LOGS/history.data')

# extract the star_age column of data
ages = h.data('star_age')

# or do it more succinctly
ages = h.star_age