"""
Returns if the single ridge in a bifurcation is 
going the same direction as the orientation (res=3)
or in the opposite (res=2)
"""

import numpy as np
from typing import Tuple
from FExtraction.p import p
from FExtraction.dist2 import dist2


def test_bifurcation(img: np.ndarray, x: int, y: int, o: float, core_x: int, core_y: int) -> Tuple[int, int, int, int, float]:
    """
    Test bifurcation direction
    
    Args:
        img: Binary image
        x, y: Bifurcation coordinates
        o: Orientation
        core_x, core_y: Core coordinates
        
    Returns:
        Tuple of (res, progress, sx, sy, angle)
    """
    iax = iay = ibx = iby = icx = icy = 0
    progress = 1
    path_len = 4

    pax = pay = pbx = pby = pcx = pcy = 0
    pao = pbo = pco = 0  # 1-8 position of 3x3 square around minutia

    # Find three branches of bifurcation
    for i in range(1, 9):
        ta, xa, ya = p(img, x, y, i)
        tb, xb, yb = p(img, x, y, i + 1)
        
        if ta > tb:
            if pao == 0:
                pao = (4 + i) if i < 5 else ((4 + i) % 9) + 1
                pax = xa
                pay = ya
            elif pbo == 0:
                pbo = (4 + i) if i < 5 else ((4 + i) % 9) + 1
                pbx = xa
                pby = ya
            else:
                pco = (4 + i) if i < 5 else ((4 + i) % 9) + 1
                pcx = xa
                pcy = ya
                break

    xaa = yaa = xbb = ybb = xcc = ycc = 0

    hist = []
    hist.append([pay, pax])
    hist.append([pby, pbx])
    hist.append([pcy, pcx])
    hist.append([y, x])
    
    stop = False
    
    while progress < path_len and not stop:
        progress += 1
        da = db = dc = 0
        
        if pao != 0:
            i = 1
            
            # Calculate CN
            cn = 0
            for ii in range(1, 9):
                t1, x_A, y_A = p(img, pax, pay, ii)
                t2, x_B, y_B = p(img, pax, pay, ii + 1)
                cn += abs(t1 - t2)
            cn = cn / 2
            
            if cn == 1 or cn == 3:
                stop = True
            
            while i < 9 and da == 0:
                ta, xa, ya = p(img, pax, pay, i)
                tz, xz, yz = p(img, pax, pay, i + 1)
                
                # Check if already visited
                found = False
                for h in hist:
                    if h[0] == ya and h[1] == xa:
                        found = True
                        break
                
                if found:
                    i += 1
                    continue
                
                if ta > tz and (xa != x or ya != y):
                    pax = xa
                    pay = ya
                    hist.append([pay, pax])
                    da = 1
                    xaa = xa
                    yaa = ya
                
                i += 1
            
            if da == 0:
                break
        
        # Similar logic for second branch
        if pbo != 0 and not stop:
            cn = 0
            for ii in range(1, 9):
                t1, x_A, y_A = p(img, pbx, pby, ii)
                t2, x_B, y_B = p(img, pbx, pby, ii + 1)
                cn += abs(t1 - t2)
            cn = cn / 2
            
            if cn == 1 or cn == 3:
                stop = True
            
            i = 1
            while i < 9 and db == 0:
                ta, xa, ya = p(img, pbx, pby, i)
                tz, xz, yz = p(img, pbx, pby, i + 1)
                
                # Check if already visited
                found = False
                for h in hist:
                    if h[0] == ya and h[1] == xa:
                        found = True
                        break
                
                if found:
                    i += 1
                    continue
                
                if ta > tz and (xa != x or ya != y):
                    pbx = xa
                    pby = ya
                    hist.append([pby, pbx])
                    db = 1
                    xbb = xa
                    ybb = ya
                
                i += 1
        
        # Similar logic for third branch
        if pco != 0 and not stop:
            cn = 0
            for ii in range(1, 9):
                t1, x_A, y_A = p(img, pcx, pcy, ii)
                t2, x_B, y_B = p(img, pcx, pcy, ii + 1)
                cn += abs(t1 - t2)
            cn = cn / 2
            
            if cn == 1 or cn == 3:
                stop = True
            
            i = 1
            while i < 9 and dc == 0:
                ta, xa, ya = p(img, pcx, pcy, i)
                tz, xz, yz = p(img, pcx, pcy, i + 1)
                
                # Check if already visited
                found = False
                for h in hist:
                    if h[0] == ya and h[1] == xa:
                        found = True
                        break
                
                if found:
                    i += 1
                    continue
                
                if ta > tz and (xa != x or ya != y):
                    pcx = xa
                    pcy = ya
                    hist.append([pcy, pcx])
                    dc = 1
                    xcc = xa
                    ycc = ya
                
                i += 1

    # Calculate distances
    d1 = np.sqrt(dist2(np.array([[xaa, yaa]]), np.array([[xbb, ybb]])))
    d2 = np.sqrt(dist2(np.array([[xaa, yaa]]), np.array([[xcc, ycc]])))
    d3 = np.sqrt(dist2(np.array([[xcc, ycc]]), np.array([[xbb, ybb]])))

    # Find the longest branch
    if d1 >= d3 and d2 >= d3:
        sx = xaa
        sy = yaa
        ind = pao
    elif d1 >= d2 and d3 >= d2:
        sx = xbb
        sy = ybb
        ind = pbo
    elif d3 >= d1 and d2 >= d1:
        sx = xcc
        sy = ycc
        ind = pco
    else:
        sx = xaa
        sy = yaa
        ind = pao

    # Calculate distances to core
    d1 = np.sqrt(dist2(np.array([[xaa, yaa]]), np.array([[core_x, core_y]])))
    d2 = np.sqrt(dist2(np.array([[xbb, ybb]]), np.array([[core_x, core_y]])))
    d3 = np.sqrt(dist2(np.array([[xcc, ycc]]), np.array([[core_x, core_y]])))

    # Find branch farthest from core
    if d1 >= d2 and d1 >= d3:
        qx = xaa
        qy = yaa
        ind = pao
    elif d2 >= d3 and d2 >= d1:
        qx = xbb
        qy = ybb
        ind = pbo
    elif d3 >= d2 and d3 >= d1:
        qx = xcc
        qy = ycc
        ind = pco
    else:
        qx = xaa
        qy = yaa
        ind = pao

    angle = np.mod(np.arctan2(y - sy, sx - x), 2 * np.pi)
    
    if qx == sx and qy == sy:
        res = 3
    else:
        res = 3

    return res, progress, sx, sy, angle


if __name__ == "__main__":
    # Example usage
    test_img = np.array([
        [0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ])
    
    res, prog, sx, sy, angle = test_bifurcation(test_img, 2, 1, 0.5, 0, 0)
    print(f"Result: {res}, Progress: {prog}, Single ridge: ({sx}, {sy}), Angle: {angle}")