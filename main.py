import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, spsolve, lsmr

image_row = 120
image_col = 120

def get_mask(img, noisy):
    threshold = 0
    if(noisy):
        print("noisy!")
        threshold = 20
    mask = np.zeros(img.shape, np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row, col] > threshold:
                mask[row, col] = 255
    return mask

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    print(mask.shape)
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.show()
    

def get_normal(imgs, lightsrc_filepath):

    I = np.reshape(imgs, (6, -1))

    L = []
    f = open(lightsrc_filepath, 'r')
    for line in f.readlines():
        nums = line[7:-2].split(",")
        light_source = [int(nums[0]), int(nums[1]), int(nums[2])]
        norm = np.linalg.norm(light_source)
        if norm != 0:
            light_source /= norm   
        L.append(light_source)

    L = np.array(L)

    '''
    N = np.linalg.solve(np.matmul(np.transpose(L), L), np.matmul(np.transpose(L), I))
    '''
    N = np.linalg.lstsq(L, I, rcond=None)[0]
    
    N = np.transpose(N)
    row_magnitudes = np.linalg.norm(N, axis=1, keepdims=True)
    for i in range(N.shape[0]):
        if(row_magnitudes[i] != 0):
            N[i] /= row_magnitudes[i]
            #print(N[i])
    return N


# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')
    plt.show()


def get_depth(N, mask):

    N = np.copy(np.reshape(N, (image_row, image_col, 3)))

    mask = np.hstack((mask, np.zeros((mask.shape[0], 1))))
    mask = np.vstack((mask, np.zeros((1, mask.shape[1]))))

    row_M = []
    col_M = []
    data_M = []
    now_row = 0

    S = image_row*image_col
    V = []
   
    
    for row in range(image_row):
        for col in range(image_col):
            pixel = row*image_col + col
            if (mask[row, col] != 0 or mask[row, col+1] != 0):    
                if(mask[row, col] != 0):
                    row_M.append(now_row)
                    col_M.append(pixel)
                    data_M.append(-1)
                if(mask[row, col+1] != 0 and col+1<image_col):
                    row_M.append(now_row)
                    col_M.append(pixel+1)
                    data_M.append(1)
                
                if(N[row, col, 2] > 0.05):
                    V.append(-1*N[row, col, 0]/N[row, col, 2])
                elif(0 < N[row, col, 2] <= 0.05):
                    V.append(V[-1])
                else:
                    V.append(0)
                now_row += 1

    for row in range(image_row):
        for col in range(image_col):
            pixel = row*image_col + col            
            if (mask[row, col] != 0 or mask[row+1, col] != 0):
                if(mask[row, col] != 0):
                    row_M.append(now_row)
                    col_M.append(pixel)
                    data_M.append(-1)
                if(mask[row+1, col] != 0 and row+1<image_row):
                    row_M.append(now_row)
                    col_M.append(pixel+image_col)
                    data_M.append(1)
                    
                if(N[row, col, 2] > 0.05):
                    V.append(N[row, col, 1]/N[row, col, 2])
                elif(0 < N[row, col, 2] <= 0.05):
                    V.append(V[-1])
                else:
                    V.append(0)
                now_row += 1

                
    '''
    for row in range(image_row):
        for col in range(image_col):
            pixel = row*image_col + col

            if (col != image_col-1):
                row_M = np.append(row_M, [pixel, pixel])
                col_M = np.append(col_M, [pixel, pixel + 1])
                data_M = np.append(data_M, [-1,1])
                
            if (row != image_row-1):
                row_M = np.append(row_M, [pixel+S, pixel+S])
                col_M = np.append(col_M, [pixel, pixel + image_col])
                data_M = np.append(data_M, [-1,1])

            if(N[row, col, 2] != 0):
                V[pixel] = (-1*N[row, col, 0]/N[row, col, 2])
                V[pixel+S] = (N[row, col, 1]/N[row, col, 2])
    '''

    print("finished step1")
    print(len(row_M),len(col_M),len(data_M))
    M = csr_matrix((data_M, (row_M, col_M)), shape = (now_row, S))
    
    '''
    print("finished step1")
    M_T = M.transpose()
    temp1 = M_T @ M
    print("finished step2")
    temp2 = M_T.dot(V)
    Z = spsolve(temp1, temp2)
    '''

    Z = lsqr(M, V)[0]
    #Z = lsmr(M, V)[0]
    
    print("finished step2")
    return Z

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(args):
    filepath = os.path.join("./test", args.case, "pic"+str(i+1)+".bmp")
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    
    if(args.noisy):
        image = cv2.GaussianBlur(image, (5, 5), 0)
    
    image_row , image_col = image.shape
    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case', type=str, default='star')
    parser.add_argument('--noisy', action="store_true")
    args = parser.parse_args()

    imgs = []
    for i in range(6):
        imgs.append(read_bmp(args))

    mask = get_mask(imgs[0], args.noisy)
    mask_visualization(mask)

    N = get_normal(imgs, os.path.join("./test", args.case, "LightSource.txt"))
    normal_visualization(N)

    Z = get_depth(N, mask)
    depth_visualization(Z)

    save_ply(Z, os.path.join("./results", args.case + ".ply"))
    show_ply(os.path.join("./results", args.case + ".ply"))

    # showing the windows of all visualization function
    #plt.show()