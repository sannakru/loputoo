import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit

# Funktsioon, mis leiab sirge, mis läbib kõik ringid

def find_line_through_points(circles,radius):
    # Leiame keskmise punkti ringide koordinaatide järgi

    #max_x = np.max(circle[0] for circle in circles)+radius
    x1=[circle[0] for circle in circles]
    max_x=np.max(x1)+radius
    y1=[circle[1] for circle in circles]
    max_y = np.max(y1)-radius
    min_x = np.min(x1)-radius
    min_y = np.min(y1)+radius
    # Sirge alguspunktina kasutame keskmist punkti
    line_start_l = (max_x, max_y)

    # Sirge lõpp-punktina kasutame sama punkti
    line_end_l = (min_x, min_y)
    circles=np.array(circles)
    miny=np.min(y1)
    where=np.where(circles[:,1]==miny)[0]
    xminmax=np.max(circles[where][0])+radius
    line_end_r=(xminmax,miny-radius)

    maxy=np.max(y1)
    where=np.where(circles[:,1]==maxy)[0]
    xmaxmin=circles[where][0][0]-radius
    line_start_r=(xmaxmin,maxy+radius)
    line_r=np.array([line_start_r,line_end_r])
    line_l=np.array([line_start_l,line_end_l])
    return line_l, line_r

# Funktsioon, mis joonistab ringid ja sirge
@jit(nopython=True)
def plot_circles_and_line(circles,new_line,new_line2,angle,radius):
    fig, ax = plt.subplots()
    # Joonistame ringid
    for circle in circles:
        circle_plot = plt.Circle((circle[0], circle[1]), radius, color='blue', fill=False)
        ax.add_artist(circle_plot)
    x=[new_line[0][0], new_line[1][0]]
    y=[new_line[0][1], new_line[1][1]]
    nx=[new_line2[0][0], new_line2[1][0]]
    ny=[new_line2[0][1], new_line2[1][1]]
    # Joonistame sirge
    plt.plot(x, y, color='red')
    plt.plot(nx, ny, color='green')
    annotation_text = f'{angle:.2f} rad'
    posx=[new_line[1][0],new_line2[1][0]]
    posy=[new_line[1][1],new_line2[1][1]]
    plt.plot(posx,posy,color='blue')
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Nurgalahutus')

    plt.xlim(min(circle[0] - 2*radius for circle in circles), max(circle[0] + radius for circle in circles))
    plt.ylim(min(circle[1] - 2*radius for circle in circles), max(circle[1] + radius for circle in circles))

    plt.grid(True)
    plt.show()



def distance_to_line(point, line):
    point = np.array(point)
    line_vector = line[1] - line[0]
    point_vector = point - line[0]
    projection = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    closest_point = line[0] + projection * line_vector
    distance = np.linalg.norm(point - closest_point)

    return distance


def move_line_to_distance(points, line, radius):
    distances = np.array([distance_to_line(point, line) for point in points])
    points = np.array(points)
    point_vectors = points - line[0]
    line_vector = line[1] - line[0]
    new_line=line
    projection = np.dot(point_vectors, line_vector) / np.dot(line_vector, line_vector)
    closest_points = np.array([(line[0]+proj*line_vector) for proj in projection])
    over_one = [index for index, number in enumerate(distances) if number >= radius+1e-13]

    neighbour_points=find_neighbour_points(points)
    neighbour_distances = np.array([distance_to_line(neighbour_points, line) 
                                    for point in neighbour_points])
    under_one = [index for index, number in enumerate(neighbour_distances) if number <= radius+1e-13]
    i=0
    while over_one!=[]:
        new_line, nlv = move_line(over_one,points,closest_points,distances,new_line,radius,0)
        new_point_vectors=points - new_line[0]
        projection = np.dot(new_point_vectors, nlv) / np.dot(nlv, nlv)
        closest_points = np.array([(new_line[0]+proj*nlv) for proj in projection])
        distances = np.array([distance_to_line(point, new_line) for point in points])
        over_one = [index for index, number in enumerate(distances) if number >= radius+1e-13]
        under_one = [index for index, number in
                      enumerate(neighbour_distances) if number <= radius+1e-13]
        i+=1
        if i>120:
            return [float('nan'),float('nan')],[float('nan'),float('nan')]
    # while under_one!=[]:
    #     new_line, nlv = move_line(under_one,points,closest_points,distances,new_line,radius,0)
    #     new_point_vectors=points - new_line[0]
    #     projection = np.dot(new_point_vectors, nlv) / np.dot(nlv, nlv)
    #     closest_points = np.array([(new_line[0]+proj*nlv) for proj in projection])
    #     distances = np.array([distance_to_line(point, new_line) for point in points])
    #     over_one = [index for index, number in enumerate(distances) if number >= radius+1e-13]
    #     under_one = [index for index, number in
    #                   enumerate(neighbour_distances) if number <= radius+1e-13]
    #     i+=1
    #     if i>120:
    #         return [float('nan'),float('nan')],[float('nan'),float('nan')]
    return new_line


def find_neighbour_points(points):
    neighbour_points=[]
    for point in points:
        if [point[0]+1,point[1]] not in points:
            neighbour=[point[0]+1,point[1]]
            neighbour_points.append(neighbour)
        if [point[0]-1,point[1]] not in points:
            neighbour=[point[0]-1,point[1]]
            neighbour_points.append(neighbour)
    return neighbour_points
     
def move_line(over_one,points,closest_points,distances,new_line,radius,n):

    # Move the line to be at least min_distance away from all points
    ind=np.random.choice(over_one)
    distance_vector=points[ind]-closest_points[ind]
    if np.linalg.norm(distance_vector) != 0:
        dis_vnorm=distance_vector/np.linalg.norm(distance_vector)
    else:
        dis_vnorm=0
    dis=distances[ind]-radius
    dis_vector=dis*dis_vnorm
    circle_point=closest_points[ind]+dis_vector

    #vaatab missugune punkt on sellele kõige lähedamal:
    dist_start=np.linalg.norm(points[ind]-new_line[0])
    dist_end=np.linalg.norm(points[ind]-new_line[1])
    #siis liigutab kas joone algust või lõppu:
    if dist_end<dist_start:
        new_line_vector_norm=(circle_point-new_line[0])/np.linalg.norm(circle_point-new_line[0])
        t=(new_line[1][1]-radius-new_line[0][1])/new_line_vector_norm[1]
        nle=np.array([t*new_line_vector_norm[0]+new_line[0][0],new_line[1][1]-radius])
        nlv=new_line[1]-new_line[0]
        new_line_=np.array([new_line[0],nle])
        return new_line_,nlv
    else:
        new_line_vector_norm=(circle_point-new_line[1])/np.linalg.norm(circle_point-new_line[1])
        t=(new_line[0][1]+radius-new_line[1][1])/new_line_vector_norm[1]
        nls=np.array([t*new_line_vector_norm[0]+new_line[1][0],new_line[0][1]+radius])
        nlv=new_line[1]-new_line[0]
        new_line_=np.array([nls,new_line[1]])
        return new_line_,nlv


def find_angle_between_lines(line1, line2):
    vector1 = np.array(line1[1]) - np.array(line1[0])
    vector2 = np.array(line2[1]) - np.array(line2[0])

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    cos_theta = dot_product / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle

def pos_resolution(x, resolution_line):
    #1,2: x | 3,4: y
    new_line1,new_line2,new_line3,new_line4= resolution_line

    pc1 = np.mean(x[0],axis=0)
    pc3 = np.mean(x[2],axis=0)
    pc5 = np.mean(x[4],axis=0)
    pc2 = np.mean(x[1],axis=0)
    pc4 = np.mean(x[3],axis=0)
    pc6 = np.mean(x[5],axis=0)
    if np.isnan(new_line1).any() or np.isnan(new_line1).any():
        pc2_res=pos_distance(pc2,new_line3,new_line4)
        pc4_res=pos_distance(pc4,new_line3,new_line4)
        pc6_res=pos_distance(pc6,new_line3,new_line4)
        pc1_res=np.nan
        pc3_res=np.nan
        pc5_res=np.nan
    elif np.isnan(new_line3).any() or np.isnan(new_line4).any():
        pc1_res=pos_distance(pc1,new_line1,new_line2)
        pc3_res=pos_distance(pc3,new_line1,new_line2)
        pc5_res=pos_distance(pc5,new_line1,new_line2)
        pc2_res=np.nan
        pc4_res=np.nan
        pc6_res=np.nan
    else:
        pc1_res=pos_distance(pc1,new_line1,new_line2)
        pc2_res=pos_distance(pc2,new_line3,new_line4)
        pc3_res=pos_distance(pc3,new_line1,new_line2)
        pc4_res=pos_distance(pc4,new_line3,new_line4)
        pc5_res=pos_distance(pc5,new_line1,new_line2)
        pc6_res=pos_distance(pc6,new_line3,new_line4)

    return pc1_res,pc2_res,pc3_res,pc4_res,pc5_res,pc6_res


def pos_distance(x,linep1,linep2):
    distance1=min(np.abs(linep1[:, 0] - x))
    distance2=min(np.abs(linep2[:, 0] - x))
    y=x[1]
    if (linep1[1][0] - linep1[0][0]!=0):
        m1 = (linep1[1][1] - linep1[0][1]) / (linep1[1][0] - linep1[0][0])
    else:
        m1=0
    b1 = linep1[0][1] - m1 * linep1[0][0]
    if (linep2[1][0] - linep2[0][0]!=0):
        m2 = (linep2[1][1] - linep2[0][1]) / (linep2[1][0] - linep2[0][0])
    else:
        m2=0
    b2 = linep2[0][1] - m2 * linep2[0][0]

    # Calculate intersection points
    if m1==0:
        x1=0
    else:
        x1 = (y - b1) / m1
    if m2==0:
        x2=0
    else:
        x2 = (y - b2) / m2

    distance = abs(x1 - x2)
    return distance
    # line_vector1 = linep1[1] - linep1[0]
    # point_vector1 = x - linep1[0]
    # projection1 = np.dot(point_vector1, line_vector1) / np.dot(line_vector1, line_vector1)
    # projection1 = np.clip(projection1, 0, 1)  # Ensure the closest point lies within the line segment
    # closest_point1 = linep1[0] + projection1 * line_vector1
    # closest_point1[1] = x[1]  # Keep the x-coordinate of the closest point the same as the given point
    # distance1 = np.linalg.norm(x - closest_point1)

    # line_vector = linep2[1] - linep2[0]
    # point_vector = x - linep2[0]
    # projection = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    # projection = np.clip(projection, 0, 1)  # Ensure the closest point lies within the line segment
    # closest_point2 = linep2[0] + projection * line_vector
    # closest_point2[1] = x[1]  # Keep the x-coordinate of the closest point the same as the given point
    # distance2 = np.linalg.norm(x - closest_point2)
    # if not closest_point1[0]<x[0]<closest_point2[0] or closest_point2[0]<x[0]<closest_point1[0]:
    #     distance= np.nan
    # else:
    #     distance= distance1+distance2
    # return distance


def resolution_lines(x):
    
    pc1 = x[0]
    pc3 = x[2]
    pc5 = x[4]
    radius=0.45

    xz_points= np.concatenate((pc1,pc3,pc5),axis=0)
    if len(xz_points)<3:
        xz_points=[np.nan,np.nan]
    # Esialgu maksimaalsed sirgete suunad
    line_l1, line_r1 = find_line_through_points(xz_points,radius)
    #Kohanda maksimaalset sirget ühes suunas:
    new_line1 = np.asarray(move_line_to_distance(xz_points, line_l1, radius=0.45))
    #Kohanda maksimaalset sirget teises suunas:
    new_line2 = np.asarray(move_line_to_distance(xz_points, line_r1, radius=0.45))

    #yz jaoks koordinaadid
    pc2 = x[1]
    pc4 = x[3]
    pc6 = x[5]

    yz_points= np.concatenate((pc2,pc4,pc6),axis=0)
    if len(yz_points)<3:
        yz_points=[np.nan,np.nan]
    line_l2, line_r2 = find_line_through_points(yz_points,radius)
    new_line3 = np.asarray(move_line_to_distance(yz_points, line_l2, radius=0.45))
    new_line4 = np.asarray(move_line_to_distance(yz_points, line_r2, radius))

    return new_line1,new_line2,new_line3,new_line4