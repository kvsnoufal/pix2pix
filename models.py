import torch
import torch.nn as nn

def Ck(in_,out_,kernel_size,stride,padding,bias,bn=True):
    b=[]
    if bn:
        b=[nn.BatchNorm2d(out_)]
    
    r=[nn.LeakyReLU(0.2,inplace=True)]
    c=[nn.Conv2d(in_,out_,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)]
    order= c + b +r
    conv=nn.Sequential(
        *order
    )
    return conv
def CDk(in_,out_,kernel_size,stride,padding,bias,bn=True,rl=True,dp=True):
    c=[nn.ConvTranspose2d(in_,out_,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)]
    d=[]
    b=[]
    r=[]
    if dp:
        d=[nn.Dropout(0.5)]
    if rl:
        r=[nn.ReLU(True)]
    if bn:
        b=[nn.BatchNorm2d(out_)]
    order=c+b+d+r
    conv=nn.Sequential(
        *order
    )
    return conv

class DiscriminatorOneChannelInput(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.d1=Ck(4,64,4,2,1,bias=False,bn=False)
        self.d2=Ck(64,128,4,2,1,False,True)
        self.d3=Ck(128,256,4,2,1,False,True)
        self.conv=nn.Conv2d(256,512,4,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(512)
        self.lr1=nn.LeakyReLU(0.2,True)
        self.last_conv=nn.Conv2d(512,1,4,1,1,bias=False)
    def forward(self,input,target):
        x=torch.cat([input,target],1)
        x=self.d1(x)
        # print(x.shape)#([1, 64, 128, 128]
        x=self.d2(x)
        # print(x.shape)#([1, 128, 64, 64])
        x=self.d3(x)
        # print(x.shape)#[1, 256, 32, 32]
        x=self.conv(x)
        # print(x.shape)#([1, 512, 31, 31])
        x=self.bn1(x)
        # print(x.shape)#([1, 512, 31, 31])
        x=self.lr1(x)
        # print(x.shape)#([1, 512, 31, 31])
        x=self.last_conv(x)#([1, 1, 30, 30])
        # print(x.shape)
        return x


class GeneratorOneChannelInput(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.e1=Ck(1,64,4,2,1,bias=False,bn=False)
        self.e2=Ck(64,128,4,2,1,bias=False,bn=True)
        self.e3=Ck(128,256,4,2,1,bias=False,bn=True)
        self.e4=Ck(256,512,4,2,1,bias=False,bn=True)
        self.e5=Ck(512,512,4,2,1,bias=False,bn=True)

        self.e6=Ck(512,512,4,2,1,bias=False,bn=True)
        self.e7=Ck(512,512,4,2,1,bias=False,bn=True)
        self.e8=Ck(512,512,4,2,1,bias=False,bn=False)
        # self.e9=Ck(512,512,4,2,1,bias=False,bn=True)

        self.d1=CDk(512,512,4,2,1,False,True,True,True)
        self.d2=CDk(1024,512,4,2,1,False,True,True,True)
        self.d3=CDk(1024,512,4,2,1,False,True,True,True)

        self.d4=CDk(1024,512,4,2,1,False,True,True,False)
        self.d5=CDk(1024,256,4,2,1,False,True,True,False)
        self.d6=CDk(512,128,4,2,1,False,True,True,False)
        self.d7=CDk(256,64,4,2,1,False,True,True,False)

        self.last_conv=nn.ConvTranspose2d(128,3,4,2,bias=True,padding=1)
        self.tanh=nn.Tanh()

    def forward(self,input):
        e1=self.e1(input)
        # print(1,e1.shape)#([1, 64, 128, 128])
        e2=self.e2(e1)
        # print(2,e2.shape)#([1, 128, 64, 64])
        e3=self.e3(e2)
        # print(3,e3.shape)#([1, 256, 32, 32])
        e4=self.e4(e3)
        # print(4,e4.shape)#([1, 512, 16, 16])
        e5=self.e5(e4)
        # print(5,e5.shape)#([1, 512, 8, 8])
        e6=self.e6(e5)
        # print(6,e6.shape)#([1, 512, 4, 4])
        e7=self.e7(e6)
        # print(7,e7.shape)#([1, 512, 2, 2])
        e8=self.e8(e7)
        # print(8,e8.shape)

        d1=self.d1(e8)
        # print(1,d1.shape)
        d1=torch.cat([d1,e7],1)
        # print(1,d1.shape)

        d2=self.d2(d1)
        # print(2,d2.shape)
        d2=torch.cat([d2,e6],1)
        # print(2,d2.shape)

        d3=self.d3(d2)
        # print(3,d3.shape)
        d3=torch.cat([d3,e5],1)
        # print(3,d3.shape)

        d4=self.d4(d3)
        # print(4,d4.shape)
        d4=torch.cat([d4,e4],1)
        # print(4,d4.shape)

        d5=self.d5(d4)
        # print(5,d5.shape)
        d5=torch.cat([d5,e3],1)
        # print(5,d5.shape)

        d6=self.d6(d5)
        # print(6,d6.shape)
        d6=torch.cat([d6,e2],1)
        # print(6,d6.shape)

        d7=self.d7(d6)
        # print(7,d7.shape)
        d7=torch.cat([d7,e1],1)
        # print(7,d7.shape)
        o=self.last_conv(d7)
        # print("o",o.shape)

        return self.tanh(o)

def init_weights(m):
    # print(type(m),m.__class__.__name__)
    if(m.__class__.__name__=="Conv2d"):
        # print("initializing.......C")
        nn.init.normal_(m.weight,0,0.02)
        if m.bias is not None:
            # print("bias init")
            nn.init.constant_(m.bias.data,0)
    if(m.__class__.__name__=="ConvTranspose2d"):
        # print("initializing.......CT")
        nn.init.normal_(m.weight,0,0.02)
        if m.bias is not None:
            # print("bias init")
            nn.init.constant_(m.bias.data,0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.d1=Ck(6,64,4,2,1,bias=False,bn=False)
        self.d2=Ck(64,128,4,2,1,False,True)
        self.d3=Ck(128,256,4,2,1,False,True)
        self.conv=nn.Conv2d(256,512,4,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(512)
        self.lr1=nn.LeakyReLU(0.2,True)
        self.last_conv=nn.Conv2d(512,1,4,1,1,bias=False)
    def forward(self,input,target):
        x=torch.cat([input,target],1)
        x=self.d1(x)
        # print(x.shape)#([1, 64, 128, 128]
        x=self.d2(x)
        # print(x.shape)#([1, 128, 64, 64])
        x=self.d3(x)
        # print(x.shape)#[1, 256, 32, 32]
        x=self.conv(x)
        # print(x.shape)#([1, 512, 31, 31])
        x=self.bn1(x)
        # print(x.shape)#([1, 512, 31, 31])
        x=self.lr1(x)
        # print(x.shape)#([1, 512, 31, 31])
        x=self.last_conv(x)#([1, 1, 30, 30])
        # print(x.shape)
        return x
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.e1=Ck(3,64,4,2,1,bias=False,bn=False)
        self.e2=Ck(64,128,4,2,1,bias=False,bn=True)
        self.e3=Ck(128,256,4,2,1,bias=False,bn=True)
        self.e4=Ck(256,512,4,2,1,bias=False,bn=True)
        self.e5=Ck(512,512,4,2,1,bias=False,bn=True)

        self.e6=Ck(512,512,4,2,1,bias=False,bn=True)
        self.e7=Ck(512,512,4,2,1,bias=False,bn=True)
        self.e8=Ck(512,512,4,2,1,bias=False,bn=False)
        # self.e9=Ck(512,512,4,2,1,bias=False,bn=True)

        self.d1=CDk(512,512,4,2,1,False,True,True,True)
        self.d2=CDk(1024,512,4,2,1,False,True,True,True)
        self.d3=CDk(1024,512,4,2,1,False,True,True,True)

        self.d4=CDk(1024,512,4,2,1,False,True,True,False)
        self.d5=CDk(1024,256,4,2,1,False,True,True,False)
        self.d6=CDk(512,128,4,2,1,False,True,True,False)
        self.d7=CDk(256,64,4,2,1,False,True,True,False)

        self.last_conv=nn.ConvTranspose2d(128,3,4,2,bias=True,padding=1)
        self.tanh=nn.Tanh()

    def forward(self,input):
        e1=self.e1(input)
        # print(1,e1.shape)#([1, 64, 128, 128])
        e2=self.e2(e1)
        # print(2,e2.shape)#([1, 128, 64, 64])
        e3=self.e3(e2)
        # print(3,e3.shape)#([1, 256, 32, 32])
        e4=self.e4(e3)
        # print(4,e4.shape)#([1, 512, 16, 16])
        e5=self.e5(e4)
        # print(5,e5.shape)#([1, 512, 8, 8])
        e6=self.e6(e5)
        # print(6,e6.shape)#([1, 512, 4, 4])
        e7=self.e7(e6)
        # print(7,e7.shape)#([1, 512, 2, 2])
        e8=self.e8(e7)
        # print(8,e8.shape)

        d1=self.d1(e8)
        # print(1,d1.shape)
        d1=torch.cat([d1,e7],1)
        # print(1,d1.shape)

        d2=self.d2(d1)
        # print(2,d2.shape)
        d2=torch.cat([d2,e6],1)
        # print(2,d2.shape)

        d3=self.d3(d2)
        # print(3,d3.shape)
        d3=torch.cat([d3,e5],1)
        # print(3,d3.shape)

        d4=self.d4(d3)
        # print(4,d4.shape)
        d4=torch.cat([d4,e4],1)
        # print(4,d4.shape)

        d5=self.d5(d4)
        # print(5,d5.shape)
        d5=torch.cat([d5,e3],1)
        # print(5,d5.shape)

        d6=self.d6(d5)
        # print(6,d6.shape)
        d6=torch.cat([d6,e2],1)
        # print(6,d6.shape)

        d7=self.d7(d6)
        # print(7,d7.shape)
        d7=torch.cat([d7,e1],1)
        # print(7,d7.shape)
        o=self.last_conv(d7)
        # print("o",o.shape)

        return self.tanh(o)                    