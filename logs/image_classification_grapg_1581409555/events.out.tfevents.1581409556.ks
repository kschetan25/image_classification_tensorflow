       ЃK"	   EзAbrain.Event:2)Л е     eНГ	ПEзA"ШЁ

conv2d_inputPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџFF*$
shape:џџџџџџџџџFF
Љ
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *ЖhЯН* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *ЖhЯ=* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
№
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 
в
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
ь
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
о
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*&
_output_shapes
:@*
T0* 
_class
loc:@conv2d/kernel
Б
conv2d/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
	container *
shape:@
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
n
conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
dtype0
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:@

conv2d/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@

conv2d/biasVarHandleOp*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d/bias
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 
_
conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
dtype0
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:@
e
conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:@

conv2d/Conv2DConv2Dconv2d_inputconv2d/Conv2D/ReadVariableOp*
paddingVALID*/
_output_shapes
:џџџџџџџџџDD@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:@

conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџDD@
a
activation/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџDD@
О
max_pooling2d/MaxPoolMaxPoolactivation/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ""@
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:ЭН*"
_class
loc:@conv2d_1/kernel

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:Э=*"
_class
loc:@conv2d_1/kernel
і
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 
к
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
є
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
ц
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
З
conv2d_1/kernelVarHandleOp* 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
t
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:@@

conv2d_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    * 
_class
loc:@conv2d_1/bias
Ѕ
conv2d_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
	container *
shape:@
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
e
conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros*
dtype0
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:@@

conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ  @*
	dilations

i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@

conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ  @
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ  @
Т
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu*/
_output_shapes
:џџџџџџџџџ@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   *"
_class
loc:@conv2d_2/kernel

.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:ЭН*"
_class
loc:@conv2d_2/kernel

.conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *:Э=*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
і
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:@@*

seed 
к
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
є
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_2/kernel
ц
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_2/kernel
З
conv2d_2/kernelVarHandleOp*
shape:@@*
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container 
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
t
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:@@

conv2d_2/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
Ѕ
conv2d_2/biasVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
e
conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros*
dtype0
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@
g
conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:@@

conv2d_2/Conv2DConv2Dmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@

conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ@
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ@
Т
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@
d
flatten/ShapeShapemax_pooling2d_2/MaxPool*
T0*
out_type0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ё
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
b
flatten/Reshape/shape/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 

flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџР

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"@     *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Й3Н*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Й3=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ц
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0*
_output_shapes
:	Р*

seed 
Ю
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
с
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
г
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
Ї
dense/kernelVarHandleOp*
_class
loc:@dense/kernel*
	container *
shape:	Р*
dtype0*
_output_shapes
: *
shared_namedense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Р

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:


dense/biasVarHandleOp*
shared_name
dense/bias*
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Р

dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
`
activation_3/SoftmaxSoftmaxdense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

activation_3_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
v
total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class

loc:@total

totalVarHandleOp*
dtype0*
_output_shapes
: *
shared_nametotal*
_class

loc:@total*
	container *
shape: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 

countVarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_namecount*
_class

loc:@count
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

metrics/acc/SqueezeSqueezeactivation_3_target*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџ
g
metrics/acc/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ

metrics/acc/ArgMaxArgMaxactivation_3/Softmaxmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
y
metrics/acc/CastCastmetrics/acc/ArgMax*

SrcT0	*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0

metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast*
T0*#
_output_shapes
:џџџџџџџџџ*
incompatible_shape_error(
z
metrics/acc/Cast_1Castmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/SumSummetrics/acc/Cast_1metrics/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0

metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
]
metrics/acc/SizeSizemetrics/acc/Cast_1*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_2Castmetrics/acc/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_2 ^metrics/acc/AssignAddVariableOp*
dtype0
 
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 

loss/activation_3_loss/CastCastactivation_3_target*

SrcT0*
Truncate( *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

DstT0	
i
loss/activation_3_loss/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
$loss/activation_3_loss/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Ј
loss/activation_3_loss/ReshapeReshapeloss/activation_3_loss/Cast$loss/activation_3_loss/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:џџџџџџџџџ
}
*loss/activation_3_loss/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
v
,loss/activation_3_loss/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,loss/activation_3_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ь
$loss/activation_3_loss/strided_sliceStridedSliceloss/activation_3_loss/Shape*loss/activation_3_loss/strided_slice/stack,loss/activation_3_loss/strided_slice/stack_1,loss/activation_3_loss/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
s
(loss/activation_3_loss/Reshape_1/shape/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
И
&loss/activation_3_loss/Reshape_1/shapePack(loss/activation_3_loss/Reshape_1/shape/0$loss/activation_3_loss/strided_slice*
N*
_output_shapes
:*
T0*

axis 
Ћ
 loss/activation_3_loss/Reshape_1Reshapedense/BiasAdd&loss/activation_3_loss/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

@loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/activation_3_loss/Reshape*
T0	*
out_type0*
_output_shapes
:
 
^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits loss/activation_3_loss/Reshape_1loss/activation_3_loss/Reshape*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
Tlabels0	*
T0
p
+loss/activation_3_loss/weighted_loss/Cast/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Yloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
і
Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:

Wloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Ю
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsh^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0*
out_type0
ѕ
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ConstConsth^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_likeFillFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:џџџџџџџџџ
к
6loss/activation_3_loss/weighted_loss/broadcast_weightsMul+loss/activation_3_loss/weighted_loss/Cast/x@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:џџџџџџџџџ
ѕ
(loss/activation_3_loss/weighted_loss/MulMul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:џџџџџџџџџ
f
loss/activation_3_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ї
loss/activation_3_loss/SumSum(loss/activation_3_loss/weighted_loss/Mulloss/activation_3_loss/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

#loss/activation_3_loss/num_elementsSize(loss/activation_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 

(loss/activation_3_loss/num_elements/CastCast#loss/activation_3_loss/num_elements*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
a
loss/activation_3_loss/Const_1Const*
dtype0*
_output_shapes
: *
valueB 

loss/activation_3_loss/Sum_1Sumloss/activation_3_loss/Sumloss/activation_3_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

loss/activation_3_loss/valueDivNoNanloss/activation_3_loss/Sum_1(loss/activation_3_loss/num_elements/Cast*
_output_shapes
: *
T0
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Z
loss/mulMul
loss/mul/xloss/activation_3_loss/value*
T0*
_output_shapes
: 
\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
j
'training/Adam/gradients/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
p
+training/Adam/gradients/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
З
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Ё
3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/activation_3_loss/value*
T0*
_output_shapes
: 

5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 

Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
м
Ntraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
З
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumSumNtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanYtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/NegNegloss/activation_3_loss/Sum_1*
T0*
_output_shapes
: 
№
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1DivNoNanGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Neg(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
љ
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2DivNoNanPtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1(loss/activation_3_loss/num_elements/Cast*
_output_shapes
: *
T0
ј
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
Д
Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1SumGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mul[training/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

Mtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Reshape_1ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

Qtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ѕ
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeReshapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeQtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0

Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 

Htraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileTileKtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 

Otraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ђ
Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeReshapeHtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileOtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Џ
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ShapeShape(loss/activation_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
Ђ
Ftraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/TileTileItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
ѓ
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
Э
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1Shape6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
ы
etraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/MulMulFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile6loss/activation_3_loss/weighted_loss/broadcast_weights*#
_output_shapes
:џџџџџџџџџ*
T0
ж
Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumSumStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Muletraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ReshapeReshapeStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
В
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1Mul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile*
T0*#
_output_shapes
:џџџџџџџџџ
м
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1gtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
а
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshape_1ReshapeUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ж
,training/Adam/gradients/gradients/zeros_like	ZerosLike`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
§
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
р
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Э
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshapetraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
ї
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMultraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimstraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Mtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
ѕ
Otraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ReshapeReshapetraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
м
@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

:training/Adam/gradients/gradients/dense/MatMul_grad/MatMulMatMulOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshapedense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџР*
transpose_a( *
transpose_b(
ј
<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
T0*
_output_shapes
:	Р*
transpose_a(*
transpose_b( 

<training/Adam/gradients/gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_2/MaxPool*
_output_shapes
:*
T0*
out_type0

>training/Adam/gradients/gradients/flatten/Reshape_grad/ReshapeReshape:training/Adam/gradients/gradients/dense/MatMul_grad/MatMul<training/Adam/gradients/gradients/flatten/Reshape_grad/Shape*/
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0
в
Jtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_2/Relumax_pooling2d_2/MaxPool>training/Adam/gradients/gradients/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@
ц
Atraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradactivation_2/Relu*/
_output_shapes
:џџџџџџџџџ@*
T0
б
Ctraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ф
=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
Х
Jtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_2/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:џџџџџџџџџ@*
	dilations

Й
Ktraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool?training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*&
_output_shapes
:@@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
о
Jtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_1/Relumax_pooling2d_1/MaxPoolJtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ  @*
T0*
data_formatNHWC*
strides

ц
Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradactivation_1/Relu*
T0*/
_output_shapes
:џџџџџџџџџ  @
б
Ctraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
Т
=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
Х
Jtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ""@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
З
Ktraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool?training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*&
_output_shapes
:@@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
и
Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradactivation/Relumax_pooling2d/MaxPoolJtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџDD@
р
?training/Adam/gradients/gradients/activation/Relu_grad/ReluGradReluGradHtraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradactivation/Relu*
T0*/
_output_shapes
:џџџџџџџџџDD@
Э
Atraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Е
;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNShapeNconv2d_inputconv2d/Conv2D/ReadVariableOp*
N* 
_output_shapes
::*
T0*
out_type0
Н
Htraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNconv2d/Conv2D/ReadVariableOp?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
paddingVALID*/
_output_shapes
:џџџџџџџџџFF*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
Ј
Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_input=training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeN:1?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@

$training/Adam/iter/Initializer/zerosConst*
value	B	 R *%
_class
loc:@training/Adam/iter*
dtype0	*
_output_shapes
: 
А
training/Adam/iterVarHandleOp*
shape: *
dtype0	*
_output_shapes
: *#
shared_nametraining/Adam/iter*%
_class
loc:@training/Adam/iter*
	container 
u
3training/Adam/iter/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
t
training/Adam/iter/AssignAssignVariableOptraining/Adam/iter$training/Adam/iter/Initializer/zeros*
dtype0	
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 

.training/Adam/beta_1/Initializer/initial_valueConst*
valueB
 *fff?*'
_class
loc:@training/Adam/beta_1*
dtype0*
_output_shapes
: 
Ж
training/Adam/beta_1VarHandleOp*'
_class
loc:@training/Adam/beta_1*
	container *
shape: *
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/beta_1
y
5training/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 

training/Adam/beta_1/AssignAssignVariableOptraining/Adam/beta_1.training/Adam/beta_1/Initializer/initial_value*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 

.training/Adam/beta_2/Initializer/initial_valueConst*
valueB
 *wО?*'
_class
loc:@training/Adam/beta_2*
dtype0*
_output_shapes
: 
Ж
training/Adam/beta_2VarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/beta_2*'
_class
loc:@training/Adam/beta_2
y
5training/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 

training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 

-training/Adam/decay/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@training/Adam/decay
Г
training/Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *$
shared_nametraining/Adam/decay*&
_class
loc:@training/Adam/decay*
	container *
shape: 
w
4training/Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/decay*
_output_shapes
: 

training/Adam/decay/AssignAssignVariableOptraining/Adam/decay-training/Adam/decay/Initializer/initial_value*
dtype0
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 
Њ
5training/Adam/learning_rate/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *o:*.
_class$
" loc:@training/Adam/learning_rate
Ы
training/Adam/learning_rateVarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/learning_rate*.
_class$
" loc:@training/Adam/learning_rate

<training/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 

"training/Adam/learning_rate/AssignAssignVariableOptraining/Adam/learning_rate5training/Adam/learning_rate/Initializer/initial_value*
dtype0

/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
Ж
/training/Adam/conv2d/kernel/m/Initializer/zerosConst* 
_class
loc:@conv2d/kernel*%
valueB@*    *
dtype0*&
_output_shapes
:@
б
training/Adam/conv2d/kernel/mVarHandleOp*.
shared_nametraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
­
>training/Adam/conv2d/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
_output_shapes
: 

$training/Adam/conv2d/kernel/m/AssignAssignVariableOptraining/Adam/conv2d/kernel/m/training/Adam/conv2d/kernel/m/Initializer/zeros*
dtype0
Й
1training/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@

-training/Adam/conv2d/bias/m/Initializer/zerosConst*
_class
loc:@conv2d/bias*
valueB@*    *
dtype0*
_output_shapes
:@
П
training/Adam/conv2d/bias/mVarHandleOp*,
shared_nametraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
Ї
<training/Adam/conv2d/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
_output_shapes
: 

"training/Adam/conv2d/bias/m/AssignAssignVariableOptraining/Adam/conv2d/bias/m-training/Adam/conv2d/bias/m/Initializer/zeros*
dtype0
Ї
/training/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_1/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_1/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_1/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/m/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_1/kernel*

index_type0*&
_output_shapes
:@@
з
training/Adam/conv2d_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@
Г
@training/Adam/conv2d_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

&training/Adam/conv2d_1/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/m1training/Adam/conv2d_1/kernel/m/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_1/bias/m/Initializer/zerosConst*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias*
valueB@*    
Х
training/Adam/conv2d_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
	container *
shape:@
­
>training/Adam/conv2d_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/m*
_output_shapes
: * 
_class
loc:@conv2d_1/bias

$training/Adam/conv2d_1/bias/m/AssignAssignVariableOptraining/Adam/conv2d_1/bias/m/training/Adam/conv2d_1/bias/m/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_2/kernel*%
valueB"      @   @   
 
7training/Adam/conv2d_2/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_2/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/m/Initializer/zeros/Const*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_2/kernel*

index_type0
з
training/Adam/conv2d_2/kernel/mVarHandleOp*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_2/kernel/m
Г
@training/Adam/conv2d_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 

&training/Adam/conv2d_2/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/m1training/Adam/conv2d_2/kernel/m/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_2/bias/m/Initializer/zerosConst*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
valueB@*    
Х
training/Adam/conv2d_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
	container *
shape:@
­
>training/Adam/conv2d_2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 

$training/Adam/conv2d_2/bias/m/AssignAssignVariableOptraining/Adam/conv2d_2/bias/m/training/Adam/conv2d_2/bias/m/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
А
>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"@     

4training/Adam/dense/kernel/m/Initializer/zeros/ConstConst*
_class
loc:@dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.training/Adam/dense/kernel/m/Initializer/zerosFill>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/m/Initializer/zeros/Const*
T0*
_class
loc:@dense/kernel*

index_type0*
_output_shapes
:	Р
Ч
training/Adam/dense/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel*
	container *
shape:	Р
Њ
=training/Adam/dense/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel*
_output_shapes
: 

#training/Adam/dense/kernel/m/AssignAssignVariableOptraining/Adam/dense/kernel/m.training/Adam/dense/kernel/m/Initializer/zeros*
dtype0
Џ
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	Р

,training/Adam/dense/bias/m/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
М
training/Adam/dense/bias/mVarHandleOp*+
shared_nametraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
Є
;training/Adam/dense/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
_output_shapes
: 

!training/Adam/dense/bias/m/AssignAssignVariableOptraining/Adam/dense/bias/m,training/Adam/dense/bias/m/Initializer/zeros*
dtype0
Є
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
Ж
/training/Adam/conv2d/kernel/v/Initializer/zerosConst* 
_class
loc:@conv2d/kernel*%
valueB@*    *
dtype0*&
_output_shapes
:@
б
training/Adam/conv2d/kernel/vVarHandleOp*.
shared_nametraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
­
>training/Adam/conv2d/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel*
_output_shapes
: 

$training/Adam/conv2d/kernel/v/AssignAssignVariableOptraining/Adam/conv2d/kernel/v/training/Adam/conv2d/kernel/v/Initializer/zeros*
dtype0
Й
1training/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@

-training/Adam/conv2d/bias/v/Initializer/zerosConst*
_class
loc:@conv2d/bias*
valueB@*    *
dtype0*
_output_shapes
:@
П
training/Adam/conv2d/bias/vVarHandleOp*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
	container *
shape:@
Ї
<training/Adam/conv2d/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
_output_shapes
: 

"training/Adam/conv2d/bias/v/AssignAssignVariableOptraining/Adam/conv2d/bias/v-training/Adam/conv2d/bias/v/Initializer/zeros*
dtype0
Ї
/training/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_1/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_1/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_1/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/v/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_1/kernel*

index_type0*&
_output_shapes
:@@
з
training/Adam/conv2d_1/kernel/vVarHandleOp*0
shared_name!training/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
Г
@training/Adam/conv2d_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

&training/Adam/conv2d_1/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/v1training/Adam/conv2d_1/kernel/v/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_1/bias/v/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Х
training/Adam/conv2d_1/bias/vVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
	container 
­
>training/Adam/conv2d_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 

$training/Adam/conv2d_1/bias/v/AssignAssignVariableOptraining/Adam/conv2d_1/bias/v/training/Adam/conv2d_1/bias/v/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_2/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_2/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

1training/Adam/conv2d_2/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/v/Initializer/zeros/Const*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_2/kernel*

index_type0
з
training/Adam/conv2d_2/kernel/vVarHandleOp*0
shared_name!training/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
Г
@training/Adam/conv2d_2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 

&training/Adam/conv2d_2/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/v1training/Adam/conv2d_2/kernel/v/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_2/bias/v/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Х
training/Adam/conv2d_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
	container *
shape:@
­
>training/Adam/conv2d_2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 

$training/Adam/conv2d_2/bias/v/AssignAssignVariableOptraining/Adam/conv2d_2/bias/v/training/Adam/conv2d_2/bias/v/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
А
>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"@     

4training/Adam/dense/kernel/v/Initializer/zeros/ConstConst*
_class
loc:@dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.training/Adam/dense/kernel/v/Initializer/zerosFill>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/v/Initializer/zeros/Const*
T0*
_class
loc:@dense/kernel*

index_type0*
_output_shapes
:	Р
Ч
training/Adam/dense/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
	container *
shape:	Р
Њ
=training/Adam/dense/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
_output_shapes
: 

#training/Adam/dense/kernel/v/AssignAssignVariableOptraining/Adam/dense/kernel/v.training/Adam/dense/kernel/v/Initializer/zeros*
dtype0
Џ
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
dtype0*
_output_shapes
:	Р*
_class
loc:@dense/kernel

,training/Adam/dense/bias/v/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
М
training/Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *+
shared_nametraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
	container *
shape:
Є
;training/Adam/dense/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
_output_shapes
: 

!training/Adam/dense/bias/v/AssignAssignVariableOptraining/Adam/dense/bias/v,training/Adam/dense/bias/v/Initializer/zeros*
dtype0
Є
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
y
%training/Adam/Identity/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
j
training/Adam/IdentityIdentity%training/Adam/Identity/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
U
training/Adam/add/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
n
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
_output_shapes
: *
T0	
m
training/Adam/CastCasttraining/Adam/add*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
t
'training/Adam/Identity_1/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_1Identity'training/Adam/Identity_1/ReadVariableOp*
T0*
_output_shapes
: 
t
'training/Adam/Identity_2/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_2Identity'training/Adam/Identity_2/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/PowPowtraining/Adam/Identity_1training/Adam/Cast*
T0*
_output_shapes
: 
i
training/Adam/Pow_1Powtraining/Adam/Identity_2training/Adam/Cast*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
N
training/Adam/SqrtSqrttraining/Adam/sub*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
e
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
h
training/Adam/mulMultraining/Adam/Identitytraining/Adam/truediv*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
l
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/Identity_1*
T0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
T0*
_output_shapes
: 
С
9training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdamResourceApplyAdamconv2d/kerneltraining/Adam/conv2d/kernel/mtraining/Adam/conv2d/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstItraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*
T0* 
_class
loc:@conv2d/kernel*
use_nesterov( *
use_locking(
Џ
7training/Adam/Adam/update_conv2d/bias/ResourceApplyAdamResourceApplyAdamconv2d/biastraining/Adam/conv2d/bias/mtraining/Adam/conv2d/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstAtraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@conv2d/bias*
use_nesterov( 
Э
;training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdamResourceApplyAdamconv2d_1/kerneltraining/Adam/conv2d_1/kernel/mtraining/Adam/conv2d_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*
use_nesterov( *
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
Л
9training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdamResourceApplyAdamconv2d_1/biastraining/Adam/conv2d_1/bias/mtraining/Adam/conv2d_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0* 
_class
loc:@conv2d_1/bias*
use_nesterov( *
use_locking(
Э
;training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdamResourceApplyAdamconv2d_2/kerneltraining/Adam/conv2d_2/kernel/mtraining/Adam/conv2d_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*
T0*"
_class
loc:@conv2d_2/kernel*
use_nesterov( *
use_locking(
Л
9training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdamResourceApplyAdamconv2d_2/biastraining/Adam/conv2d_2/bias/mtraining/Adam/conv2d_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0* 
_class
loc:@conv2d_2/bias*
use_nesterov( *
use_locking(
Џ
8training/Adam/Adam/update_dense/kernel/ResourceApplyAdamResourceApplyAdamdense/kerneltraining/Adam/dense/kernel/mtraining/Adam/dense/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@dense/kernel*
use_nesterov( 
Љ
6training/Adam/Adam/update_dense/bias/ResourceApplyAdamResourceApplyAdam
dense/biastraining/Adam/dense/bias/mtraining/Adam/dense/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense/bias*
use_nesterov( 
И
training/Adam/Adam/ConstConst8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
value	B	 R*
dtype0	*
_output_shapes
: 
x
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining/Adam/itertraining/Adam/Adam/Const*
dtype0	
ѓ
!training/Adam/Adam/ReadVariableOpReadVariableOptraining/Adam/iter'^training/Adam/Adam/AssignAddVariableOp8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp" u<     vђЖi	_EEзAJј
Й**
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Р
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

П
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
р
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype*1.15.02v1.15.0-rc3-22-g590d6eeШЁ

conv2d_inputPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџFF*$
shape:џџџџџџџџџFF
Љ
.conv2d/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d/kernel*%
valueB"         @   

,conv2d/kernel/Initializer/random_uniform/minConst* 
_class
loc:@conv2d/kernel*
valueB
 *ЖhЯН*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@conv2d/kernel*
valueB
 *ЖhЯ=*
dtype0*
_output_shapes
: 
№
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 
в
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
ь
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
о
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
Б
conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
n
conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
dtype0
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:@

conv2d/bias/Initializer/zerosConst*
_class
loc:@conv2d/bias*
valueB@*    *
dtype0*
_output_shapes
:@

conv2d/biasVarHandleOp*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d/bias
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 
_
conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
dtype0
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:@
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:@

conv2d/Conv2DConv2Dconv2d_inputconv2d/Conv2D/ReadVariableOp*
paddingVALID*/
_output_shapes
:џџџџџџџџџDD@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:@

conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџDD@
a
activation/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџDD@
О
max_pooling2d/MaxPoolMaxPoolactivation/Relu*/
_output_shapes
:џџџџџџџџџ""@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel*%
valueB"      @   @   

.conv2d_1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *:ЭН*
dtype0*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *:Э=*
dtype0*
_output_shapes
: 
і
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 
к
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
є
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
ц
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
З
conv2d_1/kernelVarHandleOp*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
t
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:@@

conv2d_1/bias/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ѕ
conv2d_1/biasVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
	container 
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
e
conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros*
dtype0
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:@@

conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*/
_output_shapes
:џџџџџџџџџ  @*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@

conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ  @
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ  @
Т
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_2/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:

.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *:ЭН

.conv2d_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *:Э=*
dtype0*
_output_shapes
: 
і
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:@@
к
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
є
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_2/kernel
ц
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
З
conv2d_2/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
t
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:@@

conv2d_2/bias/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ѕ
conv2d_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
	container *
shape:@
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
e
conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros*
dtype0
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@
g
conv2d_2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:@@

conv2d_2/Conv2DConv2Dmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@

conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ@
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ@
Т
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@
d
flatten/ShapeShapemax_pooling2d_2/MaxPool*
T0*
out_type0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ё
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
b
flatten/Reshape/shape/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:

flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџР

-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"@     

+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *Й3Н*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *Й3=*
dtype0*
_output_shapes
: 
ц
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	Р*

seed *
T0*
_class
loc:@dense/kernel
Ю
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
с
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	Р*
T0*
_class
loc:@dense/kernel
г
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
Ї
dense/kernelVarHandleOp*
shared_namedense/kernel*
_class
loc:@dense/kernel*
	container *
shape:	Р*
dtype0*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Р

dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/bias*
valueB*    


dense/biasVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_name
dense/bias*
_class
loc:@dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Р

dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
`
activation_3/SoftmaxSoftmaxdense/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

activation_3_targetPlaceholder*
dtype0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*%
shape:џџџџџџџџџџџџџџџџџџ
v
total/Initializer/zerosConst*
_class

loc:@total*
valueB
 *    *
dtype0*
_output_shapes
: 

totalVarHandleOp*
shared_nametotal*
_class

loc:@total*
	container *
shape: *
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class

loc:@count*
valueB
 *    

countVarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_namecount*
_class

loc:@count
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

metrics/acc/SqueezeSqueezeactivation_3_target*
squeeze_dims

џџџџџџџџџ*
T0*#
_output_shapes
:џџџџџџџџџ
g
metrics/acc/ArgMax/dimensionConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

metrics/acc/ArgMaxArgMaxactivation_3/Softmaxmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
y
metrics/acc/CastCastmetrics/acc/ArgMax*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ

metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast*
T0*#
_output_shapes
:џџџџџџџџџ*
incompatible_shape_error(
z
metrics/acc/Cast_1Castmetrics/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:џџџџџџџџџ
[
metrics/acc/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
{
metrics/acc/SumSummetrics/acc/Cast_1metrics/acc/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0

metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
]
metrics/acc/SizeSizemetrics/acc/Cast_1*
_output_shapes
: *
T0*
out_type0
l
metrics/acc/Cast_2Castmetrics/acc/Size*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 

!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_2 ^metrics/acc/AssignAddVariableOp*
dtype0
 
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 

metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
_output_shapes
: *
T0

loss/activation_3_loss/CastCastactivation_3_target*

SrcT0*
Truncate( *

DstT0	*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
i
loss/activation_3_loss/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
$loss/activation_3_loss/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Ј
loss/activation_3_loss/ReshapeReshapeloss/activation_3_loss/Cast$loss/activation_3_loss/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:џџџџџџџџџ
}
*loss/activation_3_loss/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
v
,loss/activation_3_loss/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
v
,loss/activation_3_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ь
$loss/activation_3_loss/strided_sliceStridedSliceloss/activation_3_loss/Shape*loss/activation_3_loss/strided_slice/stack,loss/activation_3_loss/strided_slice/stack_1,loss/activation_3_loss/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
s
(loss/activation_3_loss/Reshape_1/shape/0Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
И
&loss/activation_3_loss/Reshape_1/shapePack(loss/activation_3_loss/Reshape_1/shape/0$loss/activation_3_loss/strided_slice*
T0*

axis *
N*
_output_shapes
:
Ћ
 loss/activation_3_loss/Reshape_1Reshapedense/BiasAdd&loss/activation_3_loss/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

@loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/activation_3_loss/Reshape*
T0	*
out_type0*
_output_shapes
:
 
^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits loss/activation_3_loss/Reshape_1loss/activation_3_loss/Reshape*
T0*
Tlabels0	*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
p
+loss/activation_3_loss/weighted_loss/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

Yloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 

Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
і
Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:

Wloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Ю
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsh^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ѕ
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ConstConsth^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_likeFillFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:џџџџџџџџџ
к
6loss/activation_3_loss/weighted_loss/broadcast_weightsMul+loss/activation_3_loss/weighted_loss/Cast/x@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:џџџџџџџџџ
ѕ
(loss/activation_3_loss/weighted_loss/MulMul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:џџџџџџџџџ
f
loss/activation_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ї
loss/activation_3_loss/SumSum(loss/activation_3_loss/weighted_loss/Mulloss/activation_3_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

#loss/activation_3_loss/num_elementsSize(loss/activation_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 

(loss/activation_3_loss/num_elements/CastCast#loss/activation_3_loss/num_elements*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
a
loss/activation_3_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 

loss/activation_3_loss/Sum_1Sumloss/activation_3_loss/Sumloss/activation_3_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

loss/activation_3_loss/valueDivNoNanloss/activation_3_loss/Sum_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Z
loss/mulMul
loss/mul/xloss/activation_3_loss/value*
T0*
_output_shapes
: 
\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
j
'training/Adam/gradients/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
+training/Adam/gradients/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
З
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Ё
3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/activation_3_loss/value*
T0*
_output_shapes
: 

5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
_output_shapes
: *
T0

Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
м
Ntraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
З
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumSumNtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanYtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/NegNegloss/activation_3_loss/Sum_1*
T0*
_output_shapes
: 
№
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1DivNoNanGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Neg(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
љ
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2DivNoNanPtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
ј
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
Д
Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1SumGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mul[training/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

Mtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Reshape_1ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

Qtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ѕ
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeReshapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeQtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 

Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB 

Htraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileTileKtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 

Otraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ђ
Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeReshapeHtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileOtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Џ
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ShapeShape(loss/activation_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
Ђ
Ftraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/TileTileItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
ѓ
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
Э
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1Shape6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
ы
etraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/MulMulFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:џџџџџџџџџ
ж
Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumSumStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Muletraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ъ
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ReshapeReshapeStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
В
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1Mul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile*
T0*#
_output_shapes
:џџџџџџџџџ
м
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1gtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
а
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshape_1ReshapeUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
ж
,training/Adam/gradients/gradients/zeros_like	ZerosLike`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
§
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
р
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Э
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshapetraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ*

Tdim0
ї
training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMultraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimstraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Mtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
ѕ
Otraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ReshapeReshapetraining/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
м
@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

:training/Adam/gradients/gradients/dense/MatMul_grad/MatMulMatMulOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshapedense/MatMul/ReadVariableOp*
T0*
transpose_a( *(
_output_shapes
:џџџџџџџџџР*
transpose_b(
ј
<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
T0*
transpose_a(*
_output_shapes
:	Р*
transpose_b( 

<training/Adam/gradients/gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_2/MaxPool*
_output_shapes
:*
T0*
out_type0

>training/Adam/gradients/gradients/flatten/Reshape_grad/ReshapeReshape:training/Adam/gradients/gradients/dense/MatMul_grad/MatMul<training/Adam/gradients/gradients/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ@
в
Jtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_2/Relumax_pooling2d_2/MaxPool>training/Adam/gradients/gradients/flatten/Reshape_grad/Reshape*/
_output_shapes
:џџџџџџџџџ@*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID
ц
Atraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradactivation_2/Relu*
T0*/
_output_shapes
:џџџџџџџџџ@
б
Ctraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ф
=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
Х
Jtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_2/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
Й
Ktraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool?training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*&
_output_shapes
:@@
о
Jtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_1/Relumax_pooling2d_1/MaxPoolJtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ  @*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
ц
Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradactivation_1/Relu*
T0*/
_output_shapes
:џџџџџџџџџ  @
б
Ctraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Т
=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
Х
Jtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:џџџџџџџџџ""@
З
Ktraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool?training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*&
_output_shapes
:@@*
	dilations

и
Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradactivation/Relumax_pooling2d/MaxPoolJtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:џџџџџџџџџDD@*
T0*
data_formatNHWC*
strides

р
?training/Adam/gradients/gradients/activation/Relu_grad/ReluGradReluGradHtraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradactivation/Relu*
T0*/
_output_shapes
:џџџџџџџџџDD@
Э
Atraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Е
;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNShapeNconv2d_inputconv2d/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
Н
Htraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNconv2d/Conv2D/ReadVariableOp?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
paddingVALID*/
_output_shapes
:џџџџџџџџџFF*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
Ј
Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_input=training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeN:1?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*&
_output_shapes
:@*
	dilations


$training/Adam/iter/Initializer/zerosConst*%
_class
loc:@training/Adam/iter*
value	B	 R *
dtype0	*
_output_shapes
: 
А
training/Adam/iterVarHandleOp*
shape: *
dtype0	*
_output_shapes
: *#
shared_nametraining/Adam/iter*%
_class
loc:@training/Adam/iter*
	container 
u
3training/Adam/iter/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
t
training/Adam/iter/AssignAssignVariableOptraining/Adam/iter$training/Adam/iter/Initializer/zeros*
dtype0	
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 

.training/Adam/beta_1/Initializer/initial_valueConst*'
_class
loc:@training/Adam/beta_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ж
training/Adam/beta_1VarHandleOp*%
shared_nametraining/Adam/beta_1*'
_class
loc:@training/Adam/beta_1*
	container *
shape: *
dtype0*
_output_shapes
: 
y
5training/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 

training/Adam/beta_1/AssignAssignVariableOptraining/Adam/beta_1.training/Adam/beta_1/Initializer/initial_value*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 

.training/Adam/beta_2/Initializer/initial_valueConst*'
_class
loc:@training/Adam/beta_2*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Ж
training/Adam/beta_2VarHandleOp*
shape: *
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/beta_2*'
_class
loc:@training/Adam/beta_2*
	container 
y
5training/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 

training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 

-training/Adam/decay/Initializer/initial_valueConst*&
_class
loc:@training/Adam/decay*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
training/Adam/decayVarHandleOp*$
shared_nametraining/Adam/decay*&
_class
loc:@training/Adam/decay*
	container *
shape: *
dtype0*
_output_shapes
: 
w
4training/Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/decay*
_output_shapes
: 

training/Adam/decay/AssignAssignVariableOptraining/Adam/decay-training/Adam/decay/Initializer/initial_value*
dtype0
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 
Њ
5training/Adam/learning_rate/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@training/Adam/learning_rate*
valueB
 *o:
Ы
training/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/learning_rate*.
_class$
" loc:@training/Adam/learning_rate*
	container *
shape: 

<training/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 

"training/Adam/learning_rate/AssignAssignVariableOptraining/Adam/learning_rate5training/Adam/learning_rate/Initializer/initial_value*
dtype0

/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
Ж
/training/Adam/conv2d/kernel/m/Initializer/zerosConst*
dtype0*&
_output_shapes
:@*%
valueB@*    * 
_class
loc:@conv2d/kernel
б
training/Adam/conv2d/kernel/mVarHandleOp*.
shared_nametraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
­
>training/Adam/conv2d/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
_output_shapes
: 

$training/Adam/conv2d/kernel/m/AssignAssignVariableOptraining/Adam/conv2d/kernel/m/training/Adam/conv2d/kernel/m/Initializer/zeros*
dtype0
Й
1training/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/m*
dtype0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel

-training/Adam/conv2d/bias/m/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
П
training/Adam/conv2d/bias/mVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias
Ї
<training/Adam/conv2d/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
_output_shapes
: 

"training/Adam/conv2d/bias/m/AssignAssignVariableOptraining/Adam/conv2d/bias/m-training/Adam/conv2d/bias/m/Initializer/zeros*
dtype0
Ї
/training/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_1/kernel/m/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 

1training/Adam/conv2d_1/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/m/Initializer/zeros/Const*&
_output_shapes
:@@*
T0*

index_type0*"
_class
loc:@conv2d_1/kernel
з
training/Adam/conv2d_1/kernel/mVarHandleOp*
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
	container 
Г
@training/Adam/conv2d_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/m*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel

&training/Adam/conv2d_1/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/m1training/Adam/conv2d_1/kernel/m/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_1/bias/m/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
Х
training/Adam/conv2d_1/bias/mVarHandleOp* 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/m
­
>training/Adam/conv2d_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/m*
_output_shapes
: * 
_class
loc:@conv2d_1/bias

$training/Adam/conv2d_1/bias/m/AssignAssignVariableOptraining/Adam/conv2d_1/bias/m/training/Adam/conv2d_1/bias/m/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_2/kernel/m/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *"
_class
loc:@conv2d_2/kernel

1training/Adam/conv2d_2/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/m/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
з
training/Adam/conv2d_2/kernel/mVarHandleOp*
	container *
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel
Г
@training/Adam/conv2d_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 

&training/Adam/conv2d_2/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/m1training/Adam/conv2d_2/kernel/m/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_2/bias/m/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
Х
training/Adam/conv2d_2/bias/mVarHandleOp* 
_class
loc:@conv2d_2/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/m
­
>training/Adam/conv2d_2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 

$training/Adam/conv2d_2/bias/m/AssignAssignVariableOptraining/Adam/conv2d_2/bias/m/training/Adam/conv2d_2/bias/m/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
А
>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensorConst*
valueB"@     *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

4training/Adam/dense/kernel/m/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense/kernel

.training/Adam/dense/kernel/m/Initializer/zerosFill>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/m/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
Ч
training/Adam/dense/kernel/mVarHandleOp*
	container *
shape:	Р*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel
Њ
=training/Adam/dense/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel*
_output_shapes
: 

#training/Adam/dense/kernel/m/AssignAssignVariableOptraining/Adam/dense/kernel/m.training/Adam/dense/kernel/m/Initializer/zeros*
dtype0
Џ
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
dtype0*
_output_shapes
:	Р*
_class
loc:@dense/kernel

,training/Adam/dense/bias/m/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
М
training/Adam/dense/bias/mVarHandleOp*+
shared_nametraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
Є
;training/Adam/dense/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
_output_shapes
: 

!training/Adam/dense/bias/m/AssignAssignVariableOptraining/Adam/dense/bias/m,training/Adam/dense/bias/m/Initializer/zeros*
dtype0
Є
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
Ж
/training/Adam/conv2d/kernel/v/Initializer/zerosConst*
dtype0*&
_output_shapes
:@*%
valueB@*    * 
_class
loc:@conv2d/kernel
б
training/Adam/conv2d/kernel/vVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel*
	container 
­
>training/Adam/conv2d/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/v*
_output_shapes
: * 
_class
loc:@conv2d/kernel

$training/Adam/conv2d/kernel/v/AssignAssignVariableOptraining/Adam/conv2d/kernel/v/training/Adam/conv2d/kernel/v/Initializer/zeros*
dtype0
Й
1training/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@

-training/Adam/conv2d/bias/v/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
П
training/Adam/conv2d/bias/vVarHandleOp*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/conv2d/bias/v
Ї
<training/Adam/conv2d/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/v*
_output_shapes
: *
_class
loc:@conv2d/bias

"training/Adam/conv2d/bias/v/AssignAssignVariableOptraining/Adam/conv2d/bias/v-training/Adam/conv2d/bias/v/Initializer/zeros*
dtype0
Ї
/training/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_1/kernel/v/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *"
_class
loc:@conv2d_1/kernel

1training/Adam/conv2d_1/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/v/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
з
training/Adam/conv2d_1/kernel/vVarHandleOp*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_1/kernel/v
Г
@training/Adam/conv2d_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

&training/Adam/conv2d_1/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/v1training/Adam/conv2d_1/kernel/v/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_1/bias/v/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
Х
training/Adam/conv2d_1/bias/vVarHandleOp*.
shared_nametraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
­
>training/Adam/conv2d_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 

$training/Adam/conv2d_1/bias/v/AssignAssignVariableOptraining/Adam/conv2d_1/bias/v/training/Adam/conv2d_1/bias/v/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
О
Atraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
 
7training/Adam/conv2d_2/kernel/v/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *"
_class
loc:@conv2d_2/kernel

1training/Adam/conv2d_2/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/v/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
з
training/Adam/conv2d_2/kernel/vVarHandleOp*0
shared_name!training/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
Г
@training/Adam/conv2d_2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 

&training/Adam/conv2d_2/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/v1training/Adam/conv2d_2/kernel/v/Initializer/zeros*
dtype0
П
3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@

/training/Adam/conv2d_2/bias/v/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
Х
training/Adam/conv2d_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
	container *
shape:@
­
>training/Adam/conv2d_2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 

$training/Adam/conv2d_2/bias/v/AssignAssignVariableOptraining/Adam/conv2d_2/bias/v/training/Adam/conv2d_2/bias/v/Initializer/zeros*
dtype0
­
1training/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
А
>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensorConst*
valueB"@     *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

4training/Adam/dense/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

.training/Adam/dense/kernel/v/Initializer/zerosFill>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/v/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
Ч
training/Adam/dense/kernel/vVarHandleOp*-
shared_nametraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
	container *
shape:	Р*
dtype0*
_output_shapes
: 
Њ
=training/Adam/dense/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
_output_shapes
: 

#training/Adam/dense/kernel/v/AssignAssignVariableOptraining/Adam/dense/kernel/v.training/Adam/dense/kernel/v/Initializer/zeros*
dtype0
Џ
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
dtype0*
_output_shapes
:	Р*
_class
loc:@dense/kernel

,training/Adam/dense/bias/v/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
М
training/Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *+
shared_nametraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
	container *
shape:
Є
;training/Adam/dense/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
_output_shapes
: 

!training/Adam/dense/bias/v/AssignAssignVariableOptraining/Adam/dense/bias/v,training/Adam/dense/bias/v/Initializer/zeros*
dtype0
Є
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
y
%training/Adam/Identity/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
j
training/Adam/IdentityIdentity%training/Adam/Identity/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
U
training/Adam/add/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
n
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
T0	*
_output_shapes
: 
m
training/Adam/CastCasttraining/Adam/add*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
t
'training/Adam/Identity_1/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_1Identity'training/Adam/Identity_1/ReadVariableOp*
T0*
_output_shapes
: 
t
'training/Adam/Identity_2/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_2Identity'training/Adam/Identity_2/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/PowPowtraining/Adam/Identity_1training/Adam/Cast*
T0*
_output_shapes
: 
i
training/Adam/Pow_1Powtraining/Adam/Identity_2training/Adam/Cast*
_output_shapes
: *
T0
X
training/Adam/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
N
training/Adam/SqrtSqrttraining/Adam/sub*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
e
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
h
training/Adam/mulMultraining/Adam/Identitytraining/Adam/truediv*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *Пж3*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/Identity_1*
T0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
_output_shapes
: *
T0
С
9training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdamResourceApplyAdamconv2d/kerneltraining/Adam/conv2d/kernel/mtraining/Adam/conv2d/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstItraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0* 
_class
loc:@conv2d/kernel*
use_nesterov( 
Џ
7training/Adam/Adam/update_conv2d/bias/ResourceApplyAdamResourceApplyAdamconv2d/biastraining/Adam/conv2d/bias/mtraining/Adam/conv2d/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstAtraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@conv2d/bias*
use_nesterov( 
Э
;training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdamResourceApplyAdamconv2d_1/kerneltraining/Adam/conv2d_1/kernel/mtraining/Adam/conv2d_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*
use_nesterov( *
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
Л
9training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdamResourceApplyAdamconv2d_1/biastraining/Adam/conv2d_1/bias/mtraining/Adam/conv2d_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0* 
_class
loc:@conv2d_1/bias*
use_nesterov( *
use_locking(
Э
;training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdamResourceApplyAdamconv2d_2/kerneltraining/Adam/conv2d_2/kernel/mtraining/Adam/conv2d_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
use_nesterov( 
Л
9training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdamResourceApplyAdamconv2d_2/biastraining/Adam/conv2d_2/bias/mtraining/Adam/conv2d_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias*
use_nesterov( 
Џ
8training/Adam/Adam/update_dense/kernel/ResourceApplyAdamResourceApplyAdamdense/kerneltraining/Adam/dense/kernel/mtraining/Adam/dense/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@dense/kernel*
use_nesterov( 
Љ
6training/Adam/Adam/update_dense/bias/ResourceApplyAdamResourceApplyAdam
dense/biastraining/Adam/dense/bias/mtraining/Adam/dense/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense/bias*
use_nesterov( 
И
training/Adam/Adam/ConstConst8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
value	B	 R*
dtype0	*
_output_shapes
: 
x
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining/Adam/itertraining/Adam/Adam/Const*
dtype0	
ѓ
!training/Adam/Adam/ReadVariableOpReadVariableOptraining/Adam/iter'^training/Adam/Adam/AssignAddVariableOp8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp""с
trainable_variablesЩЦ
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08"$
	variables$$
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

training/Adam/iter:0training/Adam/iter/Assign(training/Adam/iter/Read/ReadVariableOp:0(2&training/Adam/iter/Initializer/zeros:0H

training/Adam/beta_1:0training/Adam/beta_1/Assign*training/Adam/beta_1/Read/ReadVariableOp:0(20training/Adam/beta_1/Initializer/initial_value:0H

training/Adam/beta_2:0training/Adam/beta_2/Assign*training/Adam/beta_2/Read/ReadVariableOp:0(20training/Adam/beta_2/Initializer/initial_value:0H

training/Adam/decay:0training/Adam/decay/Assign)training/Adam/decay/Read/ReadVariableOp:0(2/training/Adam/decay/Initializer/initial_value:0H
Г
training/Adam/learning_rate:0"training/Adam/learning_rate/Assign1training/Adam/learning_rate/Read/ReadVariableOp:0(27training/Adam/learning_rate/Initializer/initial_value:0H
Б
training/Adam/conv2d/kernel/m:0$training/Adam/conv2d/kernel/m/Assign3training/Adam/conv2d/kernel/m/Read/ReadVariableOp:0(21training/Adam/conv2d/kernel/m/Initializer/zeros:0
Љ
training/Adam/conv2d/bias/m:0"training/Adam/conv2d/bias/m/Assign1training/Adam/conv2d/bias/m/Read/ReadVariableOp:0(2/training/Adam/conv2d/bias/m/Initializer/zeros:0
Й
!training/Adam/conv2d_1/kernel/m:0&training/Adam/conv2d_1/kernel/m/Assign5training/Adam/conv2d_1/kernel/m/Read/ReadVariableOp:0(23training/Adam/conv2d_1/kernel/m/Initializer/zeros:0
Б
training/Adam/conv2d_1/bias/m:0$training/Adam/conv2d_1/bias/m/Assign3training/Adam/conv2d_1/bias/m/Read/ReadVariableOp:0(21training/Adam/conv2d_1/bias/m/Initializer/zeros:0
Й
!training/Adam/conv2d_2/kernel/m:0&training/Adam/conv2d_2/kernel/m/Assign5training/Adam/conv2d_2/kernel/m/Read/ReadVariableOp:0(23training/Adam/conv2d_2/kernel/m/Initializer/zeros:0
Б
training/Adam/conv2d_2/bias/m:0$training/Adam/conv2d_2/bias/m/Assign3training/Adam/conv2d_2/bias/m/Read/ReadVariableOp:0(21training/Adam/conv2d_2/bias/m/Initializer/zeros:0
­
training/Adam/dense/kernel/m:0#training/Adam/dense/kernel/m/Assign2training/Adam/dense/kernel/m/Read/ReadVariableOp:0(20training/Adam/dense/kernel/m/Initializer/zeros:0
Ѕ
training/Adam/dense/bias/m:0!training/Adam/dense/bias/m/Assign0training/Adam/dense/bias/m/Read/ReadVariableOp:0(2.training/Adam/dense/bias/m/Initializer/zeros:0
Б
training/Adam/conv2d/kernel/v:0$training/Adam/conv2d/kernel/v/Assign3training/Adam/conv2d/kernel/v/Read/ReadVariableOp:0(21training/Adam/conv2d/kernel/v/Initializer/zeros:0
Љ
training/Adam/conv2d/bias/v:0"training/Adam/conv2d/bias/v/Assign1training/Adam/conv2d/bias/v/Read/ReadVariableOp:0(2/training/Adam/conv2d/bias/v/Initializer/zeros:0
Й
!training/Adam/conv2d_1/kernel/v:0&training/Adam/conv2d_1/kernel/v/Assign5training/Adam/conv2d_1/kernel/v/Read/ReadVariableOp:0(23training/Adam/conv2d_1/kernel/v/Initializer/zeros:0
Б
training/Adam/conv2d_1/bias/v:0$training/Adam/conv2d_1/bias/v/Assign3training/Adam/conv2d_1/bias/v/Read/ReadVariableOp:0(21training/Adam/conv2d_1/bias/v/Initializer/zeros:0
Й
!training/Adam/conv2d_2/kernel/v:0&training/Adam/conv2d_2/kernel/v/Assign5training/Adam/conv2d_2/kernel/v/Read/ReadVariableOp:0(23training/Adam/conv2d_2/kernel/v/Initializer/zeros:0
Б
training/Adam/conv2d_2/bias/v:0$training/Adam/conv2d_2/bias/v/Assign3training/Adam/conv2d_2/bias/v/Read/ReadVariableOp:0(21training/Adam/conv2d_2/bias/v/Initializer/zeros:0
­
training/Adam/dense/kernel/v:0#training/Adam/dense/kernel/v/Assign2training/Adam/dense/kernel/v/Read/ReadVariableOp:0(20training/Adam/dense/kernel/v/Initializer/zeros:0
Ѕ
training/Adam/dense/bias/v:0!training/Adam/dense/bias/v/Assign0training/Adam/dense/bias/v/Read/ReadVariableOp:0(2.training/Adam/dense/bias/v/Initializer/zeros:0ўт Ф       йм2	ўй^зA*


epoch_lossBПS?Hќ       `/п#	Ѓк^зA*

	epoch_accе,?ЌЫC"       x=§	эк^зA*

epoch_val_lossЮdќ>О`!       {ьі	'л^зA*

epoch_val_accи8C?ёFЋ        )эЉP	ЊvзA*


epoch_lossR№>6ў7       QKD	ЇЊvзA*

	epoch_acc]G?4!I$       B+M	№ЊvзA*

epoch_val_loss+4к>Ѕ#       АwC	0ЊvзA*

epoch_val_accяO?ЛI
и        )эЉP	Ф\ЦзA*


epoch_lossв>`лr       QKD	g]ЦзA*

	epoch_acc\еP?ЮИЋ$       B+M	Ў]ЦзA*

epoch_val_lossBыт>іщр#       АwC	ы]ЦзA*

epoch_val_accяJ?	Ў]