import shelve

filename = 'variables.out'
my_shelf = shelve.open(filename, 'n')    # n for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        print('Error shelving: {}'.format(key))
my_shelf.close()