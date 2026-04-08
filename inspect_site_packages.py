import os
root = r'C:/Users/sepan/sparke-photo to audio/.venv/Lib/site-packages'
print('root', root)
print('exists', os.path.isdir(root))
print('matches:', [name for name in sorted(os.listdir(root)) if 'torch' in name.lower() or 'sympy' in name.lower() or name.startswith('~')])
