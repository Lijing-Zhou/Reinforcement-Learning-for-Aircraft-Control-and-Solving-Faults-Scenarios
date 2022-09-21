for key in list(globals().keys()):
 if (not key.startswith("_")) and (key !="key"):
     globals().pop(key)
del(key)     
