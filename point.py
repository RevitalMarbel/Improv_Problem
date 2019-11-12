import pygame
import sys
import random
import numpy as np
import cmath
import scipy.linalg as la
from pygame.math import Vector2
import math
import matplotlib.pyplot as plt

pygame.init()

colors = ['c','b','g','m','r','y','g','w','k','r','g','k']
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
darkBlue = (0,0,128)
white = (255,255,255)
black = (0,0,0)
pink = (255,200,200)
maxx=2000
maxy= 1000
radius=3
thickness=3
size=4
m=np.zeros((size, size),dtype=np.complex)
#print (m)
np.fill_diagonal(m, -1)
#pos_exp = 1* cmath.exp( 1j * cmath.pi/3)
#neg_exp = 1* cmath.exp( 1j * -cmath.pi/3)
pos_exp = 0.5 +(math.sqrt(3)/2)*1j
neg_exp =  0.5 -(math.sqrt(3)/2)*1j
print(pos_exp+neg_exp)
points=[]


def orient(p,q,r):
    o = np.zeros((3, 3))
    for i in range(3):
        o[0][i]=1
    o[1][0]=p.x
    o[1][1]=q.x
    o[1][2] = r.x
    o[2][0]=p.y
    o[2][1]=q.y
    o[2][2] = r.y
    print (o)
    t=np.linalg.det(o)
    if(t>0):
        return True
    else:
        return False


#generate random point in the range of the screen
def random_point(x, y, ind):
    new_x= random.randint(0+15, x-15)
    new_y=random.randint(0+15,y-15)
    v=Vector2(new_x,new_y)
    n1=random.randint(0, size - 1)
    n2=random.randint(0, size-1)
    while(n1 == ind or n2== ind or n1==n2):
        n1 = random.randint(0, size - 1)
        n2 = random.randint(0, size - 1)

    return [v,n1 ,n2]


#def arrow(screen, lcolor, tricolor, start, end, trirad):
#    pg.draw.line(screen,lcolor,start,end,2)
#    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
#    pg.draw.polygon(screen, tricolor, ((end[0]+trirad*math.sin(math.radians(rotation)), end[1]+trirad*math.cos(math.radians(rotation))), (end[0]+trirad*math.sin(math.radians(rotation-120)), end[1]+trirad*math.cos(math.radians(rotation-120))), (end[0]+trirad*math.sin(math.radians(rotation+120)), end[1]+trirad*math.cos(math.radians(rotation+120)))))'


def update_object(ticks):
    object.x += float(0.1) * ticks / 1000  # converting to float is needed
def distance(p1, p2):
    d=math.pow(p1.x-p2.x, 2)+math.pow(p1.y-p2.y,2)
    return math.sqrt(d)

#update the array point
# def update():
#     for p in points:
#         if(math.fabs(distance(p[0],points[p[1]][0]) - distance(p[0],points[p[2]][0])) >5):
#             vec= compte_angle(p[0], points[p[1]][0], points[p[2]][0])
#             p[0]=p[0]+vec

#update the array pointֿ

def update_point(p, q, r, t=0.1):
    midx=(q.x+r.x)/2
    midy=(q.y+r.y)/2

    height= math.sqrt(3)/2

    s1=Vector2( ((q.y-r.y)*height +midx , (r.x-q.x)*height+midy ))
    s2 =Vector2 (((q.y - r.y) * -height + midx, (r.x - q.x) * -height + midy))

    if(distance(p, s1)> distance(p,s2)):
        v= s2-p
    else:
        v= s1-p
    return p+v*t


def update():
    new_values=[]
    for p in points:
        new_values.append(update_point(p[0], points[p[1]][0], points[p[2]][0]))
    for i in range(len(points)):
        points[i][0]=new_values[i]


# def update():
#     for p in points:
#         #p=points[0]
#         if(math.fabs(distance(p[0],points[p[1]][0]) - distance(p[0],points[p[2]][0])) >2 or math.fabs(distance(p[0],points[p[1]][0]) - distance(points[p[1]][0],points[p[2]][0])) >2):
#             dezired_point=Vector2()
#             if(points[p[2]][0].x > points[p[1]][0].x): # p1 is in the right of p2
#                 dezired_point= compte_angle(p[0], points[p[1]][0], points[p[2]][0])
#             else:
#                 dezired_point = compte_angle(p[0], points[p[2]][0], points[p[1]][0])
#             #p_t=p[0].angle_to(dezired_point)
#             #vec=Vector2()
#             #vec.from_polar((10, p_t))
#             vec=p[0]-dezired_point
#             norm_vec=9*vec.normalize()
#             if (vec.angle_to(p[0]) > 0 ):
#                 p[0]=p[0]+norm_vec
#                 print("+")
#             else:
#                 p[0] = p[0] - norm_vec
#                 print("-")
#         else:
#             print("done")


#compute the angle to walk to from p1 to the middle between p2, p3
def compte_angle(p1, p2, p3):
    mid=[]
    d= distance(p2,p3)

    # p2_p3=p3-p2
    # p2_p3.rotate(360-60)
    # pygame.draw.circle(screen, pink, (int(p2_p3.x), int(p2_p3.y)), radius + 10, thickness + 10)
    # new=p2+p2_p3
    # print (p2_p3, "p2,p3")

    #calculate mid vector
    mid.append((p2.x+p3.x)/2)
    mid.append((p2.y+p3.y)/2)
    mid_vec=Vector2(mid)
    mid_angle=math.atan2(p3.y-p2.y,p3.x-p2.x)

    #this vector will be added to the mid vector
    perp_mid=Vector2()
    perp_mid.from_polar(((math.sqrt(3)/2)*d ,math.degrees(mid_angle)))
    print(math.degrees(mid_angle))

    #pygame.draw.circle(screen, pink, (int(perp_mid[0]), int(perp_mid[0])), radius + 10, thickness + 10)
    #ygame.draw.circle(screen, green, (int(mid_vec[0]), int(mid_vec[1])), radius + 10, thickness + 10)
    p=perp_mid.rotate(-90)
    #pygame.draw.circle(screen, blue, (int(p.x), int(p.x)), radius + 10, thickness + 10)

    new= mid_vec+p

    #print (mid)
    #pygame.draw.circle(screen, pink, (int(mid[0]), int(mid[1])), radius, thickness)
    #pygame.draw.circle(screen, black, (int(new.x), int(new.y)), radius+10, thickness+10)
    # angle=math.atan2((p1.y-mid[1]), p1.x-mid[0])
    # print(new)
    # vec = Vector2()
    # vec.from_polar((10,180+math.degrees(angle)))
    #return vec
    return new


def draw_arrow(screen, colour, start, end):
    pygame.draw.line(screen,colour,start,end,2)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pygame.draw.polygon(screen, (255, 0, 0), ((end[0]+5*math.sin(math.radians(rotation)), end[1]+5*math.cos(math.radians(rotation))), (end[0]+5*math.sin(math.radians(rotation-120)), end[1]+5*math.cos(math.radians(rotation-120))), (end[0]+5*math.sin(math.radians(rotation+120)), end[1]+5*math.cos(math.radians(rotation+120)))))


pygame.init()
paused = False
#set up the point array
for i in range(0, size):
    points.append(random_point(maxx,maxy ,i))
    print (points[i])
for i in range(size):
    p=points[i][0]
    q=points[points[i][1]][0]
    r=points[points[i][2]][0]

    t=orient(p,q, r)
    if(t):
        m[i][points[i][1]] = pos_exp
        m[i][points[i][2]] = neg_exp
    else:
        m[i][points[i][2]] = pos_exp
        m[i][points[i][1]] = neg_exp
print("m" , m)





time = pygame.time.Clock()
ticks = 0

#screen = pygame.display.set_mode((maxx,maxy))
screen = pygame.display.set_mode((maxx,maxy))
screen.fill(white)
counter=0
eigvals, eigvecs = la.eig(m)


X = [x.real for x in eigvals]
Y = [x.imag for x in eigvals]


plt.scatter(X,Y, color='red')

plt.show()

PX=[p[0].x for p in points]
PY=[p[0].y for p in points]
plt.scatter(PX,PY, color='red')


#plt.show()


ax = plt.axes()

for i in range(size):
    p = points[i][0]
    q = points[points[i][1]][0]
    r = points[points[i][2]][0]
    print(r,q)
    ax.arrow(p.x, p.y, q.x -p.x-10, q.y-p.y-10, head_width=0.05, head_length=0.05, fc='r', ec=colors[i] )
    ax.arrow(p.x, p.y, r.x-p.x-10, r.y-p.y-10, head_width=0.05, head_length=0.05, fc='r', ec=colors[i])
    ax.text(p.x, p.y,str(i) +": "+str(points[i][1])+"  "+ str(points[i][2]), fontsize=10  )
    #ax.arrow(0, 0, 50+i, 50+i, head_width=0.05, head_length=0.05, fc='k', ec='k')
plt.show()

print ("eigenvec" ,eigvecs)

for i, obj in enumerate(eigvecs):
    EVX = [x.real for x in obj]
    EVY = [x.imag for x in obj]

    plt.scatter(EVX,EVY, color=colors[i])

plt.show()

print(EVX, EVY)



#print ("orient ", t)

#print(eigvecs)

while (counter <1000000):
    counter=counter+1
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 pygame.quit(); sys.exit();
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    paused = not paused
            if not paused:
                color=20
                for p in points:
                    pygame.draw.circle(screen, (color,color,color), (int(p[0].x), int(p[0].y)), radius, thickness)
                    #draw_arrow(screen, (255,color,0),(int(p[0].x), int(p[0].y)),(int(points[p[1]][0].x),int(points[p[1]][0].y))   )
                    #draw_arrow(screen, (255,color,0), (int(p[0].x), int(p[0].y)), (int(points[p[2]][0].x), int(points[p[2]][0].y)))
                    color = color + 40
                    #pygame.draw.line(screen,blue,(int(p[0].x), int(p[0].y)),(int(points[p[1]][0].x),int(points[p[1]][0].y) ), thickness)
                    #pygame.draw.line(screen, green,(int(p[0].x), int(p[0].y)),(int(points[p[2]][0].x),int(points[p[2]][0].y) ) , thickness)
                    #update_object(ticks)
                    pygame.display.update()
                    ticks = time.tick(30)
                    counter=counter+1
            #update()

