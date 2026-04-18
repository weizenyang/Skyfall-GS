from plyfile import PlyData
import os

base = 'C:/Users/EWei/Documents/Skyfall-GS/outputs/my_scene/point_cloud'
for d in sorted(os.listdir(base), key=lambda x: int(x.split('_')[1])):
    ply = os.path.join(base, d, 'point_cloud.ply')
    if os.path.exists(ply):
        p = PlyData.read(ply)
        n = len(p.elements[0].data)
        sz = os.path.getsize(ply)
        print(f'{d:>20s}  vertices={n:>8d}  size={sz:>12,d} bytes')
