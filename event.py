# -*- coding:utf-8 -*-
import pygame
import sys
import open3d
from functools import wraps
from functools import reduce
import os
import numpy as np
import torch
from torch.autograd import Variable
import scipy.io as scio
import math

class Setting():
    def __init__(self):
        self.win_color=(240,230,140)
        self.win_size=(570,700)
        self.row_gap=90
        self.col_gap=110
        self.line_long = 150
        self.initial_loc_x=(self.line_long>>1)+self.row_gap
        self.initial_loc_y=150

        self.line_width=4
        self.line_color=(169,169,169)

        self.radius=8

        self.index=None
        #文本设置
        self.text_color = (0, 0, 0)
        self.font = pygame.font.SysFont('', 14)
        #导入是否成功标志
        self.suc_imp=-1
        #当前open3d中是否有人体标志
        self.open3d_human=False
        self.changed_human=False

class Circle():
    def __init__(self,screen,setting,circle_x,circle_y,attribute):
        self.circle_x=circle_x
        #y不变
        self.circle_y=circle_y
        self.screen=screen
        self.circle_color=(105,105,105)
        self.radius=setting.radius
        self.left_board=circle_x-(setting.line_long>>1)
        self.right_board=circle_x+(setting.line_long>>1)

        self.attribute=attribute

    def draw_circle(self):
        pygame.draw.circle(self.screen,self.circle_color,(self.circle_x,self.circle_y),
                           self.radius)

class Import():
    def __init__(self,screen,setting):
        self.butt_color = (218, 165, 32)
        self.text_rect_long=300
        self.height=35
        self.butt_long=70
        self.screen,self.setting=screen,setting
        self.font=pygame.font.SysFont('',20)

        self.text_rect=pygame.Rect(0,0,self.text_rect_long,self.height)
        self.butt_rect=pygame.Rect(0,0,self.butt_long,self.height)
        self.text_rect.center=(self.setting.win_size[0]//2-90,20)
        self.butt_rect.top=self.text_rect.top
        self.butt_rect.left=self.text_rect.right+5
        self.imp_click = pygame.Rect(0,0,self.butt_long-4,self.height-4)
        self.imp_click.center=self.butt_rect.center
        self.imp=self.font.render('Import',True,self.setting.text_color,self.butt_color)
        self.imp_rect=self.imp.get_rect()
        self.imp_rect.center=self.butt_rect.center
        self.butt1_rect=pygame.Rect(0,0,self.butt_long,self.height)
        self.butt1_rect.centerx,self.butt1_rect.bottom=self.setting.win_size[0]//2,self.setting.win_size[1]-20
        self.chan=self.font.render('Change',True,self.setting.text_color,self.butt_color)
        self.chan_rect=self.chan.get_rect()
        self.chan_rect.center=self.butt1_rect.center
        self.chan_click=pygame.Rect(0,0,self.butt_long-4,self.height-4)
        self.chan_click.center=self.butt1_rect.center

        self.click=None
        self.s,self.s_rects=[],[]
        self.ch,self.key,self.constant,self.last=None,None,False,None

        self.bu=5
        self.bu_rect1 = pygame.Rect(0, 0, self.bu, self.height)
        self.bu_rect1.top,self.bu_rect1.left=self.text_rect.top,self.text_rect.left
        self.bu_rect2=pygame.Rect(0,0,self.bu,self.height)
        self.bu_rect2.top,self.bu_rect2.right=self.text_rect.top,self.text_rect.right

        self.mark,self.mark_long,self.front_long=0,self.text_rect.left+self.bu,0
        self.behind_long=0
        self.lb,self.rb=self.text_rect.left+self.bu,self.text_rect.right-self.bu

        self.prompt1=pygame.font.SysFont('stix',10).render('The file does not exist!',True,(255,0,0),setting.win_color)
        self.prompt1_rect=self.prompt1.get_rect()
        self.prompt1_rect.left=self.text_rect.left
        self.prompt1_rect.top=self.text_rect.bottom+2
        self.prompt2=pygame.font.SysFont('stix',10).render('Successfully imported!',True,(0,191,255),setting.win_color)
        self.prompt2_rect=self.prompt2.get_rect()
        self.prompt2_rect.left=self.text_rect.left
        self.prompt2_rect.top=self.text_rect.bottom+2
        self.prompt3=pygame.font.SysFont('stix',10).render('Importing...',True,(0,191,255),setting.win_color)
        self.prompt3_rect=self.prompt3.get_rect()
        self.prompt3_rect.left=self.text_rect.left
        self.prompt3_rect.top=self.text_rect.bottom+2

    def add_chr(self,chr):
        new_chr=self.font.render(chr,True,self.setting.text_color,self.butt_color)
        #tempfunc1=lambda :self.text_rect.left+5 if self.mark==0  else self.s_rects[self.mark-1][1].right
        new_chr_rect=new_chr.get_rect()
        new_chr_rect.center=self.text_rect.center
        #new_chr_rect.left=tempfunc1()
        self.s.insert(self.mark,chr)
        self.s_rects.insert(self.mark,(new_chr,new_chr_rect))
        self.mark+=1
        self.front_long+=new_chr_rect.width
        if self.mark_long + new_chr.get_width() <= self.rb:
            self.mark_long=self.mark_long+new_chr.get_width()

    def pop_chr(self):
        self.mark-=1
        if self.behind_long<=self.rb-self.mark_long:
            if self.front_long-self.s_rects[self.mark][1].width<=self.mark_long-self.lb:
                if self.front_long<=self.mark_long-self.lb:
                    self.mark_long-=self.s_rects[self.mark][1].width
                else:
                    self.mark_long=reduce(lambda x,y:x+y[1].width,self.s_rects[:self.mark],self.lb)
        elif self.mark_long-self.s_rects[self.mark][1].width<self.lb:
            self.mark_long=self.lb
        else:
             self.mark_long-=self.s_rects[self.mark][1].width
        self.front_long -= self.s_rects[self.mark][1].width
        del self.s[self.mark]
        del self.s_rects[self.mark]

    def mark_move_left(self):
        if self.mark>0:
            self.mark-=1
            self.front_long -= self.s_rects[self.mark][1].width
            self.behind_long+=self.s_rects[self.mark][1].width
            if self.mark_long - self.s_rects[self.mark][1].width >= self.lb:
                self.mark_long-=self.s_rects[self.mark][1].width

    def mark_move_right(self):
        if self.mark<len(self.s):
            if self.mark_long + self.s_rects[self.mark][1].width<=self.rb:
                self.mark_long+= self.s_rects[self.mark][1].width
            self.front_long += self.s_rects[self.mark][1].width
            self.behind_long-=self.s_rects[self.mark][1].width
            self.mark+=1

    def get_content(self):
        long,mark=self.mark_long,self.mark
        while long<self.rb and mark<len(self.s):
            long+=self.s_rects[mark][1].width
            self.s_rects[mark][1].right=long
            self.screen.blit(self.s_rects[mark][0], self.s_rects[mark][1])
            mark+=1
        if long>self.rb:
            pygame.draw.rect(self.screen,self.butt_color,self.bu_rect2)
            if long>self.text_rect.right:
                rect=pygame.Rect(0,0,long-self.text_rect.right,self.height)
                rect.top,rect.left=self.text_rect.top,self.text_rect.right
                pygame.draw.rect(self.screen,self.setting.win_color,rect)
        long,mark=self.mark_long,self.mark-1
        while long>self.lb and mark>=0:
            long-=self.s_rects[mark][1].width
            self.s_rects[mark][1].left=long
            self.screen.blit(self.s_rects[mark][0], self.s_rects[mark][1])
            mark-=1
        if long<self.lb:
            pygame.draw.rect(self.screen,self.butt_color,self.bu_rect1)
            if long<self.text_rect.left:
                rect=pygame.Rect(0,0,self.text_rect.left-long,self.height)
                rect.top,rect.right=self.text_rect.top,self.text_rect.left
                pygame.draw.rect(self.screen,self.setting.win_color,rect)

    def draw(self):
        pygame.draw.rect(self.screen,self.butt_color,self.text_rect)
        #for i in range(len(self.s_rects)):
            #self.screen.blit(self.s_rects[i][0],self.s_rects[i][1])
        self.get_content()
        pygame.draw.rect(self.screen, self.butt_color, self.butt_rect)
        pygame.draw.rect(self.screen,self.butt_color,self.butt1_rect)
        self.screen.blit(self.imp, self.imp_rect)
        self.screen.blit(self.chan,self.chan_rect)
        if self.click==0:
            pygame.draw.rect(self.screen,(105,105,105),self.imp_click,1)
        elif self.click==1:
            pygame.draw.rect(self.screen,(105,105,105),self.chan_click,1)
        if (pygame.time.get_ticks()//600)&1==0 or not self.key is None:
            pygame.draw.line(self.screen,self.setting.text_color,(self.mark_long,7),(self.mark_long,33),1)
        #pygame.draw.line(self.screen, self.setting.text_color, (self.lb, 7), (self.lb, 33),1)
        #pygame.draw.line(self.screen, self.setting.text_color, (self.rb, 7), (self.rb, 33), 1)

class Export():
    def __init__(self,screen,setting,im):
        self.screen,self.setting,self.im=screen,setting,im
        self.font = pygame.font.SysFont('stix', 18)
        self.font1 = pygame.font.SysFont('stix', 10)
        self.lit_butt_long=70
        self.lit_height=35

        self.butt1_rect=pygame.Rect(0,0,self.lit_butt_long,self.lit_height)
        self.butt1_rect.left,self.butt1_rect.top=im.butt_rect.right+5,im.butt_rect.top
        self.export = self.font.render('Export', True, setting.text_color, im.butt_color)
        self.export_rect = self.export.get_rect()
        self.export_rect.center = self.butt1_rect.center

        self.ex_bg=pygame.Rect(0,0,400,100)
        self.ex_bg.center=(self.setting.win_size[0]//2,self.setting.win_size[1]//2-60)
        self.text=pygame.Rect(0,0,350,35)
        self.text.left,self.text.top=self.ex_bg.left+25,self.ex_bg.top+10
        self.butt2_rect=pygame.Rect(0,0,im.butt_long,im.height)
        self.butt2_rect.bottom,self.butt2_rect.right=self.ex_bg.bottom-10,self.ex_bg.right-25
        self.butt3_rect=pygame.Rect(0,0,im.butt_long,im.height)
        self.butt3_rect.bottom,self.butt3_rect.right=self.ex_bg.bottom-10,self.butt2_rect.left-15
        self.confirm=self.font.render('Confirm',True,setting.text_color,setting.win_color)
        self.confirm_rect=self.confirm.get_rect()
        self.cancel=self.font.render('Cancel',True,setting.text_color,setting.win_color)
        self.cancel_rect=self.cancel.get_rect()
        self.confirm_rect.center,self.cancel_rect.center=self.butt3_rect.center,self.butt2_rect.center

        self.hint1=self.font1.render('Stored Successfully!',True,(255,0,0),im.butt_color)
        self.hint1_rect=self.hint1.get_rect()
        self.hint1_rect.left,self.hint1_rect.top=self.text.left,self.text.bottom+2
        self.hint2=self.font1.render('The path does not exis!',True,(0,191,255),im.butt_color)
        self.hint2_rect=self.hint2.get_rect()
        self.hint2_rect.left,self.hint2_rect.top=self.text.left,self.text.bottom+2
        self.hint=0

        self.state=False

        self.s, self.s_rects = [], []
        self.ch, self.key, self.constant, self.last = None, None, False, None

        self.bu = 5
        self.bu_rect1 = pygame.Rect(0, 0, self.bu, im.height)
        self.bu_rect1.top, self.bu_rect1.left = self.text.top, self.text.left
        self.bu_rect2 = pygame.Rect(0, 0, self.bu, im.height)
        self.bu_rect2.top, self.bu_rect2.right = self.text.top, self.text.right

        self.mark, self.mark_long, self.front_long = 0, self.text.left + self.bu, 0
        self.behind_long = 0
        self.lb, self.rb = self.text.left + self.bu, self.text.right - self.bu

    def add_chr(self,chr):
        new_chr=self.font.render(chr,True,self.setting.text_color,self.setting.win_color)
        new_chr_rect=new_chr.get_rect()
        new_chr_rect.center=self.text.center
        self.s.insert(self.mark,chr)
        self.s_rects.insert(self.mark,(new_chr,new_chr_rect))
        self.mark+=1
        self.front_long+=new_chr_rect.width
        if self.mark_long + new_chr.get_width() <= self.rb:
            #说明字符串长度小于文本框长度
            self.mark_long=self.mark_long+new_chr.get_width()
        print(self.s)

    def get_content(self):
        long,mark=self.mark_long,self.mark
        while long<self.rb and mark<len(self.s):
            long+=self.s_rects[mark][1].width
            self.s_rects[mark][1].right=long
            self.screen.blit(self.s_rects[mark][0], self.s_rects[mark][1])
            mark+=1
        if long>self.rb:
            pygame.draw.rect(self.screen,self.setting.win_color,self.bu_rect2)
            if long>self.text.right:
                rect=pygame.Rect(0,0,long-self.text.right,self.text.height)
                rect.top,rect.left=self.text.top,self.text.right
                pygame.draw.rect(self.screen,self.im.butt_color,rect)
        long,mark=self.mark_long,self.mark-1
        while long>self.lb and mark>=0:
            long-=self.s_rects[mark][1].width
            self.s_rects[mark][1].left=long
            self.screen.blit(self.s_rects[mark][0], self.s_rects[mark][1])
            mark-=1
        if long<self.lb:
            pygame.draw.rect(self.screen,self.setting.win_color,self.bu_rect1)
            if long<self.text.left:
                rect=pygame.Rect(0,0,self.text.left-long,self.text.height)
                rect.top,rect.right=self.text.top,self.text.left
                pygame.draw.rect(self.screen,self.im.butt_color,rect)

    def draw(self):
        pygame.draw.rect(self.screen,self.im.butt_color,self.butt1_rect)
        self.screen.blit(self.export,self.export_rect)
        if self.state:
            pygame.draw.rect(self.screen,self.im.butt_color,self.ex_bg)
            pygame.draw.rect(self.screen,self.setting.win_color,self.text)
            pygame.draw.rect(self.screen,self.setting.win_color,self.butt2_rect)
            pygame.draw.rect(self.screen,self.setting.win_color,self.butt3_rect)
            self.screen.blit(self.confirm,self.confirm_rect)
            self.screen.blit(self.cancel,self.cancel_rect)
            self.get_content()
            if (pygame.time.get_ticks() // 600) & 1 == 0:
                pygame.draw.line(self.screen,self.setting.text_color,(self.mark_long,self.text.bottom-5),
                                 (self.mark_long,self.text.top+5),1)
            if self.hint==1:
                self.screen.blit(self.hint1,self.hint1_rect)
            elif self.hint==2:
                self.screen.blit(self.hint2,self.hint2_rect)

class Model(torch.nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6,n_output):
        super(Model,self).__init__()
        self.layer1=torch.nn.Linear(n_input,n_hidden1)
        self.layer2=torch.nn.Linear(n_hidden1,n_hidden2)
        self.layer3=torch.nn.Linear(n_hidden2,n_hidden3)
        self.layer4=torch.nn.Linear(n_hidden3,n_hidden4)
        self.layer5=torch.nn.Linear(n_hidden4,n_hidden5)
        self.layer6=torch.nn.Linear(n_hidden5,n_hidden6)
        self.layer7=torch.nn.Linear(n_hidden6,n_output)

    def forward(self, x):
        x=torch.tanh(self.layer1(x))
        x=torch.tanh(self.layer2(x))
        x=torch.tanh(self.layer3(x))
        x=torch.tanh(self.layer4(x))
        x=torch.tanh(self.layer5(x))
        x=torch.tanh(self.layer6(x))
        x=self.layer7(x)
        return x

class Attribute():
    def __init__(self):
        self.attributes=None
        self.interval={'WaistThickness':(0.18,0.99,14),
                       #0.59
                       'Waist':(0.1,2.04,15),
                       #0.3,2.04
                       'ChestWide':(0.28,0.58,16),
                       #0.58
                       'ChestThickness':(0.14,0.44,12),
                       #0.44
                       'Chest':(0.1,1.83,18),
                       #0.88,1.83
                       'Hip':(0.16,0.91,11),
                       #0.31
                       'Shoulder':(0.22,1.67,13),
                       #0.67
                       'Thinness':(0.1,1.836,22),
                       #0.686,0.836
                       'Plumpness':(0.09,0.67,26),
                       #0.67
                       'Elbows':(0.42,0.87,28)}
                        #0.87
        self.nums=30
    def match(self,file):
        name=os.path.splitext(os.path.basename(file))[0]
        with open('Vectors.txt','r') as f:
            for line in f.readlines():
                line=line.split()
                if line[0]==name:
                    self.attributes=list(map(np.float32,line[1:]))
                    break
            f.close()

def check_keydown_events(event,im,ex):
    if not ex.state:
        im.ch,im.key=None,None
        if event.key==pygame.K_BACKSPACE:
            if  im.mark>0:
                im.pop_chr()
                im.key=event.key
        elif event.key==pygame.K_LEFT:
            im.mark_move_left()
            im.key=event.key
        elif event.key==pygame.K_RIGHT:
            im.mark_move_right()
            im.key=event.key
        elif len(event.unicode)==0:
            return
        elif ord(event.unicode)>=32 and ord(event.unicode)<=126:
            im.add_chr(event.unicode)
            im.ch,im.key=event.unicode,event.key
            #im.key=event.key
        im.last = pygame.time.get_ticks()
        im.constant=False
    else:
        ex.ch,ex.key=None,None
        if len(event.unicode)==0:
            return
        elif ord(event.unicode)>=32 and ord(event.unicode)<=126:
            ex.add_chr(event.unicode)
            ex.ch,ex.key=event.unicode,event.key
        ex.last = pygame.time.get_ticks()
        ex.constant = False

def check_keyup_events(event,im):
    if event.key==im.key:
        im.ch, im.key, im.constant, im.last = None, None, False, None
    elif (event.key==pygame.K_LSHIFT or event.key==pygame.K_RSHIFT) and not im.ch is None:
        im.ch=im.ch.lower()

def check_hold_on_event(im):
    if im.key==pygame.K_BACKSPACE:
        if im.mark > 0:
            if im.constant:
                im.pop_chr()
            elif pygame.time.get_ticks() - im.last >= 400:
                im.constant = True
                im.pop_chr()
    elif im.key==pygame.K_LEFT:
        if im.constant:
            im.mark_move_left()
        elif pygame.time.get_ticks()-im.last>=400:
            im.constant=True
            im.mark_move_left()
    elif im.key==pygame.K_RIGHT:
        if im.constant:
            im.mark_move_right()
        elif pygame.time.get_ticks() - im.last >= 400:
            im.constant = True
            im.mark_move_right()
    elif not im.key is None:
        if im.constant:
            im.add_chr(im.ch)
        elif pygame.time.get_ticks() - im.last >= 400:
            im.constant = True
            im.add_chr(im.ch)

def obj_to_mesh(file,mesh):
    vertices,triangles=[],[]
    with open(file,'r') as f:
        for line in f.readlines():
            if line[0]=='v':
                vertices.append(list(map(float,line.split()[1:])))
            elif line[0]=='f':
                triangles.append(list(map(lambda x:int(x)-1,line.split()[1:])))
        f.close()
    #mesh[0].vertices=open3d.Vector3dVector(np.array(vertices))
    mesh[0].triangles=open3d.Vector3iVector(np.array(triangles))
    mesh[1].triangles=mesh[0].triangles
    mesh[2].triangles=mesh[0].triangles
    new=np.array(mesh[0].vertices)
    old=np.array(vertices)
    diff=new-old
    #print(np.max(diff[:,0]),np.max(diff[:,1]),np.max(diff[:,2]))
    #print(np.mean(diff[:,0]),np.mean(diff[:,1]),np.mean(diff[:,2]))
    temp,temp1=[],[]
    for i in range(len(diff)):
        temp.append(math.sqrt(diff[i][0]**2+diff[i][1]**2+diff[i][2]**2))
        temp1.append(math.sqrt(old[i][0] ** 2 + old[i][1] ** 2 + old[i][2] ** 2))
    #print(np.mean(np.array(temp)),np.max(np.array(temp)))
    #print(np.mean(np.array(temp1)), np.max(np.array(temp1)))

def check_button(setting,cg,im,ex,attr,x,y,model,mesh,vis):
    if ex.state:
        if ex.butt3_rect.collidepoint(x,y):
            store(ex,mesh)
        elif ex.butt2_rect.collidepoint(x,y):
            ex.state=False
    if im.butt_rect.collidepoint(x,y):
        im.click=0
        file=''.join(im.s)
        if not os.path.exists(file) or not file.endswith('.obj'):
            setting.suc_imp=0
        else:
            attr.match(file)
            temp = Variable(torch.from_numpy(np.array(attr.attributes, dtype=np.float32)))
            result = np.array(model(temp).data.numpy(),dtype=float)
            mesh[0].vertices=open3d.Vector3dVector(get_vertices(result))
            setting.suc_imp=1
        return True
    elif im.butt1_rect.collidepoint(x,y):
        im.click=1
        cir_reflict_attr(setting,cg,attr,model,mesh,vis)
    elif ex.butt1_rect.collidepoint(x,y):
        ex.state=True
    return False
    #print(attr.attributes)

def check_events(screen,setting,cg,im,ex,HOLD_ON,attr,model,mesh,vis):
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()
        elif event.type==pygame.MOUSEBUTTONDOWN:
            mx,my=pygame.mouse.get_pos()
            if check_button(setting,cg,im,ex,attr,mx,my,model,mesh,vis):
                continue
            #直接找寻块区域,无需遍历
            indx,indy=None,None
            temp=setting.line_long+setting.row_gap
            if mx%temp-setting.row_gap>=0 and mx//temp<=2:
                indx=mx//temp
            else:
                continue
            my1=my-(setting.initial_loc_y+setting.radius-setting.col_gap)
            temp1=setting.col_gap
            temp2=setting.col_gap-setting.radius*2
            #print(mx,my,my1,temp1,temp2)
            if my1%temp1-temp2>0 and my1//temp1<5:
                indy=my1//temp1
            else:
                continue
            #print(indx,indy)
            temp=indx*5+indy
            #print(abs(mx-cg[temp].circle_x),abs(my-cg[temp].circle_y))
            if temp in cg.keys() and abs(mx-cg[temp].circle_x)<=setting.radius and abs(my-cg[temp].circle_y)<=setting.\
                    radius:
                setting.index=temp
            else:
                continue
        elif event.type==pygame.MOUSEBUTTONUP:
            setting.index=None
            im.click=None
        elif event.type==pygame.KEYDOWN:
            check_keydown_events(event,im,ex)
        elif event.type==pygame.KEYUP:
            check_keyup_events(event,im)
        elif event.type==HOLD_ON:
            check_hold_on_event(im)
        if not setting.index is None:
            mx,my=pygame.mouse.get_pos()
            if mx>cg[setting.index].left_board and mx<=cg[setting.index].right_board:
                cg[setting.index].circle_x=mx
            elif mx<cg[setting.index].left_board:
                cg[setting.index].circle_x=cg[setting.index].left_board
            else:
                cg[setting.index].circle_x = cg[setting.index].right_board

def update_screen(screen,setting,lg,cg,tg,im,ex):
    screen.fill(setting.win_color)
    im.draw()
    for i in range(len(lg)):
        pygame.draw.rect(screen,setting.line_color,lg[i])
        screen.blit(tg[i][0],tg[i][1])
    for key in cg.keys():
        cg[key].draw_circle()
    if setting.suc_imp==0:
        screen.blit(im.prompt1,im.prompt1_rect)
    elif setting.suc_imp==1:
        screen.blit(im.prompt3, im.prompt3_rect)
    elif setting.suc_imp==2:
        screen.blit(im.prompt2, im.prompt2_rect)
    ex.draw()
    pygame.display.flip()

def store(ex,mesh):
    ss=''.join(ex.s)
    if os.path.exists(os.path.dirname(ss)):
        ex.hint=1
        if len(np.array(mesh[1].vertices))>0:
            np.savetxt(ss,np.vstack((np.array(mesh[1].vertices),np.array(mesh[1].triangles))))
        else:
            np.savetxt(ss, np.vstack((np.array(mesh[0].vertices), np.array(mesh[0].triangles))))
    else:
        ex.hint=2

def con_rect_cir(screen,setting,rg,cg):
    '''content = ['Height', 'Head', 'Trunk', 'UpperBody', 'LowerBody', 'BigLeg', 'LittleLeg', 'Arm', 'BigArm',
                    'LittleArm', 'Shoulder', 'Hip', 'Nipples', 'WaistWide', 'WaistThickness', 'Waist', 'ChestWide',
                    'ChestThickness', 'Chest', 'Head/Height', 'UpperBody/Height', 'LowerBody/Height', 'Upper/Lower',
                    'BigArm/Arm', 'LittleArm/Arm', 'LittleArm/BigArm', 'Foots', 'Knees', 'Elbows', 'Wrists']'''
    content = ['WaistThickness', 'Waist', 'ChestWide', 'ChestThickness', 'Chest', 'Hip', 'Shoulder', 'Thinness',
               'Plumpness', 'Elbows']
    x,y=setting.initial_loc_x,None
    for i in range(len(content)):
        rect = pygame.Rect(0, 0, setting.line_long, setting.line_width)
        if i%5==0 and i>0:
            x=(i//5+1)*(setting.row_gap+setting.line_long)-setting.line_long//2
            y=setting.initial_loc_y
        else:
            y=setting.initial_loc_y+(i%5)*setting.col_gap
        rect.center=(x,y)
        rg.append(rect)
        new_circle=Circle(screen,setting,x,y,content[i])
        cg[i]=new_circle

def con_text(setting,rg,tg):
    '''content = ['Height', 'Head', 'Trunk', 'UpperBody', 'LowerBody', 'BigLeg', 'LittleLeg', 'Arm', 'BigArm',
                    'LittleArm', 'Shoulder', 'Hip', 'Nipples', 'WaistWide', 'WaistThickness', 'Waist', 'ChestWide',
                    'ChestThickness', 'Chest', 'Head/Height', 'UpperBody/Height', 'LowerBody/Height', 'Upper/Lower',
                    'BigArm/Arm', 'LittleArm/Arm', 'LittleArm/BigArm', 'Foots', 'Knees', 'Elbows', 'Wrists']'''
    content=['WaistThickness','Waist','ChestWide','ChestThickness','Chest','Hip','Shoulder','Thinness','Plumpness','Elbows']
    for i in range(len(content)):
        word=setting.font.render(content[i],True,setting.text_color,setting.win_color)
        rect=word.get_rect()
        rect.left=rg[i].left
        rect.bottom=rg[i].top-10
        tg.append((word,rect))
    #for zi in pygame.font.get_fonts():
        #print(zi)

def attr_reflict_cir(setting,cg,at):
    gap=setting.line_long
    tempfunc=lambda k:int(round(gap*((at.attributes[at.interval[cg[k].attribute][2]]-at.interval[cg[k].attribute][0])/
                                     (at.interval[cg[k].attribute][1]-at.interval[cg[k].attribute][0]))))
    #temp=[]
    for i in range(len(cg)):
        cg[i].circle_x=cg[i].left_board+tempfunc(i)
        #temp.append(cg[i].circle_x)
    #print(temp)

def cir_reflict_attr(setting,cg,at,model,mesh,vis):
    tempfunc=lambda k:(at.interval[cg[k].attribute][1]-at.interval[cg[k].attribute][0])*\
                      ((cg[k].circle_x-cg[k].left_board)/setting.line_long)
    temp=[None]*30
    for i in range(len(cg)):
        temp[at.interval[cg[i].attribute][2]]=at.interval[cg[i].attribute][0]+tempfunc(i)
    temp1,name=[],[]
    for i in range(len(temp)):
        if temp[i] is None:
            temp[i]=at.attributes[i]
        else:
            temp1.append(temp[i])
    print(temp1)
    temp=Variable(torch.from_numpy(np.array(temp,dtype=np.float32)))
    result=get_vertices(np.array(model(temp).data.numpy(),dtype=float))
    #加1.2为了移位置
    result[:,0]+=1.2
    mesh[1].vertices=open3d.Vector3dVector(result)
    if setting.changed_human==False:
        vis.add_geometry(mesh[1])
        setting.changed_human=True
        ctr = vis.get_view_control()
        ctr.rotate(0.0, -500.0)
    mesh[1].compute_vertex_normals()
    mesh[1].paint_uniform_color([1, 0.706, 0])

def get_vertices(coefs):
    bsapce = scio.loadmat('bspace_train_8_12.mat')
    meanbody = bsapce['mean'].squeeze()
    pcadata = bsapce['pca'][0:100]
    body = (meanbody + coefs.dot(pcadata)).reshape(-1, 3)
    return body

def run_window(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        vis = open3d.Visualizer()
        vis.create_window(window_name='Human', width=570, height=700, left=720, top=180, visible=True)

        pygame.init()
        setting=Setting()
        screen=pygame.display.set_mode(setting.win_size)
        pygame.display.set_caption('roll')

        mesh=[open3d.TriangleMesh(),open3d.TriangleMesh(),open3d.TriangleMesh()]
        attr = Attribute()

        line_group,cir_group,text_group=[],{},list()
        con_rect_cir(screen,setting,line_group,cir_group)
        con_text(setting,line_group,text_group)
        im=Import(screen,setting)
        ex=Export(screen,setting,im)

        HOLD_ON=pygame.USEREVENT+1 #自定义事件
        pygame.time.set_timer(HOLD_ON,60)

        model=Model(30,32,42,54,66,78,90,100)
        model.load_state_dict(torch.load('human_net.pkl'))
        while True:
            check_events(screen,setting,cir_group,im,ex,HOLD_ON,attr,model,mesh,vis)
            update_screen(screen,setting,line_group,cir_group,text_group,im,ex)
            if setting.suc_imp==1:
                obj_to_mesh(''.join(im.s),mesh)
                vis.add_geometry(mesh[0])
                mesh[0].compute_vertex_normals()
                ctr = vis.get_view_control()
                ctr.rotate(0.0, -500.0)
                attr_reflict_cir(setting,cir_group,attr)
                setting.suc_imp=2
            vis.update_geometry()
            vis.poll_events()
    return decorated

