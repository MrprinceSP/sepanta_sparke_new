import pkg_resources
print('has packaging:', hasattr(pkg_resources, 'packaging'))
print('matches:', [n for n in dir(pkg_resources) if 'packag' in n.lower()])
