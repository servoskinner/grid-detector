from matplotlib import pyplot as plt
from grid_detector import Grid

def plot_grid(points, grid : Grid, label=None):
    """
    Draw a grid over given set of points.
    """
    origin, u_basis, v_basis, dims = grid.origin, grid.basis_u, grid.basis_v, grid.dims
    plt.figure(figsize=(8, 8))
    plt.xlim(-0.1, 1.1)  
    plt.ylim(-0.1, 1.1)  

    for i in range(dims[0]):
        plt.plot([origin[0] + i * u_basis[0], origin[0] + i * u_basis[0] + v_basis[0] * (dims[1] - 1)], 
                 [origin[1] + i * u_basis[1], origin[1] + i * u_basis[1] + v_basis[1] * (dims[1] - 1)], 
                 color='seagreen', linewidth=2, zorder=1, alpha=0.4)
    
    for j in range(dims[1]):
        plt.plot([origin[0] + j * v_basis[0], origin[0] + j * v_basis[0] + u_basis[0] * (dims[0] - 1)], 
                 [origin[1] + j * v_basis[1], origin[1] + j * v_basis[1] + u_basis[1] * (dims[0] - 1)], 
                 color='gray', linewidth=2, zorder=1, alpha=0.6)
        
    for point in points:
        plt.plot(point[0], point[1], 'o', color='gray', markersize=16, markeredgewidth=6, fillstyle='none', zorder=2)
    
    plt.arrow(origin[0], origin[1], u_basis[0], u_basis[1], 
              linewidth=4, head_width=0.05, head_length=0.05, fc='red', ec='red', zorder=3)
    plt.arrow(origin[0], origin[1], v_basis[0], v_basis[1], 
              linewidth=4, head_width=0.05, head_length=0.05, fc='blue', ec='blue', zorder=3)
    
    plt.plot(origin[0], origin[1], 'o', color='darkgreen', markersize=10, markeredgewidth=6, fillstyle='full', zorder=4)
    
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig("grid.jpg" if label==None else f"{label}.jpg")


N_DUMPS = 0
def dump_points(points, label=None):
    """
    Dump a set of points as image for debug purposes.
    """
    global N_DUMPS
    plt.figure(figsize=(8, 8))
    plt.xlim(-0.25, 1.25)  
    plt.ylim(-0.25, 1.25)  
        
    for point in points:
        plt.plot(point[0], point[1], 'o', color='gray', markersize=16, markeredgewidth=6, fillstyle='none', zorder=2)
    
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    if label != None:
        plt.savefig(f"{label}.jpg")
    else:
        plt.savefig(f"debug{N_DUMPS}.jpg")
        N_DUMPS += 1