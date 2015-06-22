## @package utils
# General utilites

##
# Update defaults arguments with the provided argumetns
def update_defaults(ipArgs, defArgs):
	'''
		ipArgs : input (provided) arguments
		defArgs: default arguments
	'''
	for key in ipArgs.keys():
    assert defArgs.has_key(key), 'Key not found: %s' % key
    defArgs[key] = ipArgs[key]
  return defArgs
	


