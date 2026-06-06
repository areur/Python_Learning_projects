import numpy as np
import matplotlib.pyplot as mlt
from matplotlib.widgets import Slider, Button

print("son")

fig, plots = mlt.subplots(1,2,figsize=(20,5))
fig.subplots_adjust(left=0.25,bottom=0.25)

currMatrix = "Shear"
currFactor = 1
ShearYAxis = False

axtext = fig.add_axes((0.6,0.05,0.1,0.075))
axtext.axis('off')
textbox = axtext.text(0,0,currMatrix,fontsize = 12)

# Default Transformation Matrices
def rotationMatrix(theta):
    currMatrix = "Rotation"
    rad = np.radians(theta)
    return np.array([[np.cos(rad),-np.sin(rad)], [np.sin(rad),np.cos(rad)]])

def shearMatrix(factor: float, y: bool = False):
    currMatrix = "Shear"
    if y:
        mat = np.array([[1,factor], [0,1]])
    else:
        mat = np.array([[1,0], [factor,1]])
    return mat

CanonBasis = np.array([[1,0],[0,1]])
transMatrix = shearMatrix(2,True)
transformedBasis = np.dot(transMatrix,CanonBasis)


def updatePlot():
    global transformedBasis, PlotPostTransform
    transformedBasis = np.dot(transMatrix,CanonBasis)
    PlotPostTransform.set_UVC(5*transformedBasis[:,0],5*transformedBasis[:,1])

def updateButton(event):
    global currMatrix, rotAxSlider, rotSlider, SheAxSlider, sheSlider, SheAxButton, shearToggle
    if currMatrix == "Shear":
        currMatrix = "Rotation"
        rotAxSlider.set_visible(True)
        rotSlider.active = True
        SheAxSlider.set_visible(False)
        sheSlider.active = False
        SheAxButton.set_visible(False)
        shearToggle.active = False
        updateSlider(rotSlider.val)
    else: 
        currMatrix = "Shear"
        SheAxSlider.set_visible(True)
        sheSlider.active = True
        SheAxButton.set_visible(True)
        shearToggle.active = True
        rotAxSlider.set_visible(False)
        rotSlider.active = False
        updateSlider(sheSlider.val)
    textbox.set_text(currMatrix)
    fig.canvas.draw_idle()

def toggleShearAxis(event):
    global ShearYAxis
    ShearYAxis = not ShearYAxis
    updateSlider(sheSlider.val)

def updateSlider(value):
    global currFactor, currMatrix, transMatrix 
    currFactor = value
    if currMatrix == "Shear":
        transMatrix = shearMatrix(currFactor,ShearYAxis)
    else: 
        transMatrix = rotationMatrix(currFactor)
    updatePlot()
    fig.canvas.draw_idle()

axbutton = fig.add_axes((0.7,0.05,0.1,0.075))
buttonMatrix = Button(axbutton,'Change Transformation Type')
buttonMatrix.on_clicked(updateButton)

rotAxSlider = fig.add_axes((0.1,0.3,0.03,0.65))
SheAxSlider = fig.add_axes((0.1,0.3,0.03,0.65))

rotSlider = Slider(
    ax=rotAxSlider,
    label="Rotation Angle",
    valmin=0,
    valmax=360,
    valinit=45,
    orientation="vertical"
)
rotSlider.on_changed(updateSlider)

sheSlider = Slider(
    ax=SheAxSlider,
    label="Shear Factor",
    valmin=0,
    valmax=10,
    valinit=0,
    orientation="vertical"
)
sheSlider.on_changed(updateSlider)

SheAxSlider.set_visible(True)
sheSlider.active = True
rotAxSlider.set_visible(False)
rotSlider.active = False
SheAxButton = fig.add_axes((0.1,0.1,0.1,0.1))
shearToggle = Button(SheAxButton,'Toggle Shear Axis')
shearToggle.on_clicked(toggleShearAxis)

origin = np.array([[0,0],[0,0]])
PlotPreTransform = plots[0].quiver(*origin, 5*CanonBasis[:,0],5*CanonBasis[:,1],color=['r','b','g'],scale=1, angles='xy',scale_units='xy')
plots[0].set_xlim(-10,10)
plots[0].set_ylim(-10,10)
plots[0].grid(True)

PlotPostTransform = plots[1].quiver(*origin, 5*transformedBasis[:,0],5*transformedBasis[:,1],color=['r','b','g'],scale=1, angles='xy',scale_units='xy')
plots[1].set_xlim(-10,10)
plots[1].set_ylim(-10,10)
plots[1].grid(True)
mlt.show()