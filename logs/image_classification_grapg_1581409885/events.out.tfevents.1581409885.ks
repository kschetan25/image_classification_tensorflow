       �K"	  @����Abrain.Event:2&��rm:     ���	kEu����A"��

conv2d_inputPlaceholder*
dtype0*/
_output_shapes
:���������FF*$
shape:���������FF
�
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�hϽ* 
_class
loc:@conv2d/kernel
�
,conv2d/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�h�=* 
_class
loc:@conv2d/kernel
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:@*

seed 
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
�
conv2d/kernelVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
	container 
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
�
conv2d/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
�
conv2d/biasVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d/bias*
_class
loc:@conv2d/bias
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
�
conv2d/Conv2DConv2Dconv2d_inputconv2d/Conv2D/ReadVariableOp*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������DD@*
	dilations
*
T0
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:@
�
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������DD@
a
activation/ReluReluconv2d/BiasAdd*/
_output_shapes
:���������DD@*
T0
�
max_pooling2d/MaxPoolMaxPoolactivation/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������""@
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:͓�*"
_class
loc:@conv2d_1/kernel
�
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_1/kernel*
seed2 *
dtype0*&
_output_shapes
:@@*

seed 
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_1/kernel
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
�
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@
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
�
conv2d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
�
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
�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*/
_output_shapes
:���������  @*
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
�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������  @
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:���������  @
�
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@
�
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
�
.conv2d_2/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓�*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
�
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:@@
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
conv2d_2/kernelVarHandleOp*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel
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
�
conv2d_2/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
�
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
�
conv2d_2/Conv2DConv2Dmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
paddingVALID*/
_output_shapes
:���������@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@
�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
�
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@
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
flatten/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
b
flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
�
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"@     *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *�3�*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *�3=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0*
_output_shapes
:	�
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
�
dense/kernelVarHandleOp*
shape:	�*
dtype0*
_output_shapes
: *
shared_namedense/kernel*
_class
loc:@dense/kernel*
	container 
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
:	�
�
dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@dense/bias
�

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
:	�
�
dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
�
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:���������*
T0
`
activation_3/SoftmaxSoftmaxdense/BiasAdd*
T0*'
_output_shapes
:���������
�
activation_3_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
�
totalVarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_nametotal*
_class

loc:@total
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
: *
valueB
 *    *
_class

loc:@count
�
countVarHandleOp*
dtype0*
_output_shapes
: *
shared_namecount*
_class

loc:@count*
	container *
shape: 
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
�
metrics/acc/SqueezeSqueezeactivation_3_target*
squeeze_dims

���������*
T0*#
_output_shapes
:���������
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxactivation_3/Softmaxmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
y
metrics/acc/CastCastmetrics/acc/ArgMax*

SrcT0	*
Truncate( *#
_output_shapes
:���������*

DstT0
�
metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast*#
_output_shapes
:���������*
incompatible_shape_error(*
T0
z
metrics/acc/Cast_1Castmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/SumSummetrics/acc/Cast_1metrics/acc/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
�
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
�
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_2 ^metrics/acc/AssignAddVariableOp*
dtype0
�
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
�
loss/activation_3_loss/CastCastactivation_3_target*

SrcT0*
Truncate( *0
_output_shapes
:������������������*

DstT0	
i
loss/activation_3_loss/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
$loss/activation_3_loss/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
�
loss/activation_3_loss/ReshapeReshapeloss/activation_3_loss/Cast$loss/activation_3_loss/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:���������
}
*loss/activation_3_loss/strided_slice/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:
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
�
$loss/activation_3_loss/strided_sliceStridedSliceloss/activation_3_loss/Shape*loss/activation_3_loss/strided_slice/stack,loss/activation_3_loss/strided_slice/stack_1,loss/activation_3_loss/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
s
(loss/activation_3_loss/Reshape_1/shape/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
&loss/activation_3_loss/Reshape_1/shapePack(loss/activation_3_loss/Reshape_1/shape/0$loss/activation_3_loss/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
 loss/activation_3_loss/Reshape_1Reshapedense/BiasAdd&loss/activation_3_loss/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
@loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/activation_3_loss/Reshape*
T0	*
out_type0*
_output_shapes
:
�
^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits loss/activation_3_loss/Reshape_1loss/activation_3_loss/Reshape*
T0*?
_output_shapes-
+:���������:������������������*
Tlabels0	
p
+loss/activation_3_loss/weighted_loss/Cast/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Yloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
�
Wloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsh^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ConstConsth^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_likeFillFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:���������
�
6loss/activation_3_loss/weighted_loss/broadcast_weightsMul+loss/activation_3_loss/weighted_loss/Cast/x@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
(loss/activation_3_loss/weighted_loss/MulMul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
f
loss/activation_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/activation_3_loss/SumSum(loss/activation_3_loss/weighted_loss/Mulloss/activation_3_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
#loss/activation_3_loss/num_elementsSize(loss/activation_3_loss/weighted_loss/Mul*
_output_shapes
: *
T0*
out_type0
�
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
�
loss/activation_3_loss/Sum_1Sumloss/activation_3_loss/Sumloss/activation_3_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
loss/activation_3_loss/valueDivNoNanloss/activation_3_loss/Sum_1(loss/activation_3_loss/num_elements/Cast*
_output_shapes
: *
T0
O

loss/mul/xConst*
valueB
 *  �?*
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
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
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
 *  �?*
dtype0*
_output_shapes
: 
�
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/activation_3_loss/value*
T0*
_output_shapes
: 
�
5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ntraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumSumNtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanYtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/NegNegloss/activation_3_loss/Sum_1*
T0*
_output_shapes
: 
�
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1DivNoNanGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Neg(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2DivNoNanPtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2*
_output_shapes
: *
T0
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1SumGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mul[training/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
Mtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Reshape_1ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Qtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeReshapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeQtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB 
�
Htraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileTileKtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Const*
_output_shapes
: *

Tmultiples0*
T0
�
Otraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeReshapeHtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileOtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ShapeShape(loss/activation_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
�
Ftraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/TileTileItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
�
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1Shape6loss/activation_3_loss/weighted_loss/broadcast_weights*
_output_shapes
:*
T0*
out_type0
�
etraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/MulMulFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
�
Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumSumStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Muletraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ReshapeReshapeStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1Mul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1gtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshape_1ReshapeUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
,training/Adam/gradients/gradients/zeros_like	ZerosLike`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshape�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*0
_output_shapes
:������������������
�
Mtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ShapeShapedense/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Otraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ReshapeReshape�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
�
:training/Adam/gradients/gradients/dense/MatMul_grad/MatMulMatMulOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshapedense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
<training/Adam/gradients/gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_2/MaxPool*
T0*
out_type0*
_output_shapes
:
�
>training/Adam/gradients/gradients/flatten/Reshape_grad/ReshapeReshape:training/Adam/gradients/gradients/dense/MatMul_grad/MatMul<training/Adam/gradients/gradients/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
Jtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_2/Relumax_pooling2d_2/MaxPool>training/Adam/gradients/gradients/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@
�
Atraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradactivation_2/Relu*/
_output_shapes
:���������@*
T0
�
Ctraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
N* 
_output_shapes
::*
T0*
out_type0
�
Jtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_2/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������@*
	dilations
*
T0
�
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
�
Jtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_1/Relumax_pooling2d_1/MaxPoolJtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:���������  @*
T0*
data_formatNHWC*
strides

�
Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradactivation_1/Relu*
T0*/
_output_shapes
:���������  @
�
Ctraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Jtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������""@*
	dilations
*
T0
�
Ktraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool?training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
paddingVALID*&
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
use_cudnn_on_gpu(
�
Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradactivation/Relumax_pooling2d/MaxPoolJtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:���������DD@*
T0*
data_formatNHWC*
strides

�
?training/Adam/gradients/gradients/activation/Relu_grad/ReluGradReluGradHtraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradactivation/Relu*
T0*/
_output_shapes
:���������DD@
�
Atraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNShapeNconv2d_inputconv2d/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Htraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNconv2d/Conv2D/ReadVariableOp?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*/
_output_shapes
:���������FF*
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
�
Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_input=training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeN:1?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@*
	dilations

�
$training/Adam/iter/Initializer/zerosConst*
value	B	 R *%
_class
loc:@training/Adam/iter*
dtype0	*
_output_shapes
: 
�
training/Adam/iterVarHandleOp*#
shared_nametraining/Adam/iter*%
_class
loc:@training/Adam/iter*
	container *
shape: *
dtype0	*
_output_shapes
: 
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
�
.training/Adam/beta_1/Initializer/initial_valueConst*
valueB
 *fff?*'
_class
loc:@training/Adam/beta_1*
dtype0*
_output_shapes
: 
�
training/Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *%
shared_nametraining/Adam/beta_1*'
_class
loc:@training/Adam/beta_1*
	container *
shape: 
y
5training/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
�
training/Adam/beta_1/AssignAssignVariableOptraining/Adam/beta_1.training/Adam/beta_1/Initializer/initial_value*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
�
.training/Adam/beta_2/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*'
_class
loc:@training/Adam/beta_2
�
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
�
training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
�
-training/Adam/decay/Initializer/initial_valueConst*
valueB
 *    *&
_class
loc:@training/Adam/decay*
dtype0*
_output_shapes
: 
�
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
�
5training/Adam/learning_rate/Initializer/initial_valueConst*
valueB
 *o�:*.
_class$
" loc:@training/Adam/learning_rate*
dtype0*
_output_shapes
: 
�
training/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/learning_rate*.
_class$
" loc:@training/Adam/learning_rate*
	container *
shape: 
�
<training/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
�
"training/Adam/learning_rate/AssignAssignVariableOptraining/Adam/learning_rate5training/Adam/learning_rate/Initializer/initial_value*
dtype0
�
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
�
/training/Adam/conv2d/kernel/m/Initializer/zerosConst* 
_class
loc:@conv2d/kernel*%
valueB@*    *
dtype0*&
_output_shapes
:@
�
training/Adam/conv2d/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
	container *
shape:@
�
>training/Adam/conv2d/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/m*
_output_shapes
: * 
_class
loc:@conv2d/kernel
�
$training/Adam/conv2d/kernel/m/AssignAssignVariableOptraining/Adam/conv2d/kernel/m/training/Adam/conv2d/kernel/m/Initializer/zeros*
dtype0
�
1training/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@
�
-training/Adam/conv2d/bias/m/Initializer/zerosConst*
_class
loc:@conv2d/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training/Adam/conv2d/bias/mVarHandleOp*,
shared_nametraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
<training/Adam/conv2d/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/m*
_output_shapes
: *
_class
loc:@conv2d/bias
�
"training/Adam/conv2d/bias/m/AssignAssignVariableOptraining/Adam/conv2d/bias/m-training/Adam/conv2d/bias/m/Initializer/zeros*
dtype0
�
/training/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
�
Atraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_1/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
7training/Adam/conv2d_1/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1training/Adam/conv2d_1/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/m/Initializer/zeros/Const*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_1/kernel*

index_type0
�
training/Adam/conv2d_1/kernel/mVarHandleOp*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_1/kernel/m
�
@training/Adam/conv2d_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/m*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
�
&training/Adam/conv2d_1/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/m1training/Adam/conv2d_1/kernel/m/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@
�
/training/Adam/conv2d_1/bias/m/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training/Adam/conv2d_1/bias/mVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias
�
>training/Adam/conv2d_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
�
$training/Adam/conv2d_1/bias/m/AssignAssignVariableOptraining/Adam/conv2d_1/bias/m/training/Adam/conv2d_1/bias/m/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
�
Atraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_2/kernel*%
valueB"      @   @   
�
7training/Adam/conv2d_2/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1training/Adam/conv2d_2/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/m/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_2/kernel*

index_type0*&
_output_shapes
:@@
�
training/Adam/conv2d_2/kernel/mVarHandleOp*
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
	container 
�
@training/Adam/conv2d_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/m*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel
�
&training/Adam/conv2d_2/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/m1training/Adam/conv2d_2/kernel/m/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@
�
/training/Adam/conv2d_2/bias/m/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training/Adam/conv2d_2/bias/mVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
	container 
�
>training/Adam/conv2d_2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 
�
$training/Adam/conv2d_2/bias/m/AssignAssignVariableOptraining/Adam/conv2d_2/bias/m/training/Adam/conv2d_2/bias/m/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
�
>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense/kernel*
valueB"@     *
dtype0*
_output_shapes
:
�
4training/Adam/dense/kernel/m/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *    
�
.training/Adam/dense/kernel/m/Initializer/zerosFill>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/m/Initializer/zeros/Const*
_output_shapes
:	�*
T0*
_class
loc:@dense/kernel*

index_type0
�
training/Adam/dense/kernel/mVarHandleOp*-
shared_nametraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
: 
�
=training/Adam/dense/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/m*
_output_shapes
: *
_class
loc:@dense/kernel
�
#training/Adam/dense/kernel/m/AssignAssignVariableOptraining/Adam/dense/kernel/m.training/Adam/dense/kernel/m/Initializer/zeros*
dtype0
�
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	�
�
,training/Adam/dense/bias/m/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/dense/bias/mVarHandleOp*
dtype0*
_output_shapes
: *+
shared_nametraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
	container *
shape:
�
;training/Adam/dense/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/m*
_output_shapes
: *
_class
loc:@dense/bias
�
!training/Adam/dense/bias/m/AssignAssignVariableOptraining/Adam/dense/bias/m,training/Adam/dense/bias/m/Initializer/zeros*
dtype0
�
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
�
/training/Adam/conv2d/kernel/v/Initializer/zerosConst*
dtype0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel*%
valueB@*    
�
training/Adam/conv2d/kernel/vVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel
�
>training/Adam/conv2d/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
$training/Adam/conv2d/kernel/v/AssignAssignVariableOptraining/Adam/conv2d/kernel/v/training/Adam/conv2d/kernel/v/Initializer/zeros*
dtype0
�
1training/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/v*
dtype0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel
�
-training/Adam/conv2d/bias/v/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@conv2d/bias*
valueB@*    
�
training/Adam/conv2d/bias/vVarHandleOp*,
shared_nametraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
<training/Adam/conv2d/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/v*
_output_shapes
: *
_class
loc:@conv2d/bias
�
"training/Adam/conv2d/bias/v/AssignAssignVariableOptraining/Adam/conv2d/bias/v-training/Adam/conv2d/bias/v/Initializer/zeros*
dtype0
�
/training/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
�
Atraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_1/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
7training/Adam/conv2d_1/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1training/Adam/conv2d_1/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/v/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_1/kernel*

index_type0*&
_output_shapes
:@@
�
training/Adam/conv2d_1/kernel/vVarHandleOp*
	container *
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel
�
@training/Adam/conv2d_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
&training/Adam/conv2d_1/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/v1training/Adam/conv2d_1/kernel/v/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@
�
/training/Adam/conv2d_1/bias/v/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training/Adam/conv2d_1/bias/vVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
	container 
�
>training/Adam/conv2d_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
�
$training/Adam/conv2d_1/bias/v/AssignAssignVariableOptraining/Adam/conv2d_1/bias/v/training/Adam/conv2d_1/bias/v/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
�
Atraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_2/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
7training/Adam/conv2d_2/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1training/Adam/conv2d_2/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/v/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_2/kernel*

index_type0*&
_output_shapes
:@@
�
training/Adam/conv2d_2/kernel/vVarHandleOp*0
shared_name!training/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
�
@training/Adam/conv2d_2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/v*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel
�
&training/Adam/conv2d_2/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/v1training/Adam/conv2d_2/kernel/v/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@
�
/training/Adam/conv2d_2/bias/v/Initializer/zerosConst*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias*
valueB@*    
�
training/Adam/conv2d_2/bias/vVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias
�
>training/Adam/conv2d_2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 
�
$training/Adam/conv2d_2/bias/v/AssignAssignVariableOptraining/Adam/conv2d_2/bias/v/training/Adam/conv2d_2/bias/v/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/v*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense/kernel*
valueB"@     *
dtype0*
_output_shapes
:
�
4training/Adam/dense/kernel/v/Initializer/zeros/ConstConst*
_class
loc:@dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
.training/Adam/dense/kernel/v/Initializer/zerosFill>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/v/Initializer/zeros/Const*
_output_shapes
:	�*
T0*
_class
loc:@dense/kernel*

index_type0
�
training/Adam/dense/kernel/vVarHandleOp*-
shared_nametraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
: 
�
=training/Adam/dense/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/v*
_output_shapes
: *
_class
loc:@dense/kernel
�
#training/Adam/dense/kernel/v/AssignAssignVariableOptraining/Adam/dense/kernel/v.training/Adam/dense/kernel/v/Initializer/zeros*
dtype0
�
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	�
�
,training/Adam/dense/bias/v/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/dense/bias/vVarHandleOp*+
shared_nametraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
�
;training/Adam/dense/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
_output_shapes
: 
�
!training/Adam/dense/bias/v/AssignAssignVariableOptraining/Adam/dense/bias/v,training/Adam/dense/bias/v/Initializer/zeros*
dtype0
�
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
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
T0	*
_output_shapes
: 
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
 *  �?*
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
 *  �?
e
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
h
training/Adam/mulMultraining/Adam/Identitytraining/Adam/truediv*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
l
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/Identity_1*
_output_shapes
: *
T0
Z
training/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
_output_shapes
: *
T0
�
9training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdamResourceApplyAdamconv2d/kerneltraining/Adam/conv2d/kernel/mtraining/Adam/conv2d/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstItraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0* 
_class
loc:@conv2d/kernel*
use_nesterov( 
�
7training/Adam/Adam/update_conv2d/bias/ResourceApplyAdamResourceApplyAdamconv2d/biastraining/Adam/conv2d/bias/mtraining/Adam/conv2d/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstAtraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@conv2d/bias*
use_nesterov( 
�
;training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdamResourceApplyAdamconv2d_1/kerneltraining/Adam/conv2d_1/kernel/mtraining/Adam/conv2d_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*
T0*"
_class
loc:@conv2d_1/kernel*
use_nesterov( *
use_locking(
�
9training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdamResourceApplyAdamconv2d_1/biastraining/Adam/conv2d_1/bias/mtraining/Adam/conv2d_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_1/bias*
use_nesterov( 
�
;training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdamResourceApplyAdamconv2d_2/kerneltraining/Adam/conv2d_2/kernel/mtraining/Adam/conv2d_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*
T0*"
_class
loc:@conv2d_2/kernel*
use_nesterov( *
use_locking(
�
9training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdamResourceApplyAdamconv2d_2/biastraining/Adam/conv2d_2/bias/mtraining/Adam/conv2d_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0* 
_class
loc:@conv2d_2/bias
�
8training/Adam/Adam/update_dense/kernel/ResourceApplyAdamResourceApplyAdamdense/kerneltraining/Adam/dense/kernel/mtraining/Adam/dense/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@dense/kernel*
use_nesterov( 
�
6training/Adam/Adam/update_dense/bias/ResourceApplyAdamResourceApplyAdam
dense/biastraining/Adam/dense/bias/mtraining/Adam/dense/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0*
_class
loc:@dense/bias
�
training/Adam/Adam/ConstConst8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
value	B	 R*
dtype0	*
_output_shapes
: 
x
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining/Adam/itertraining/Adam/Adam/Const*
dtype0	
�
!training/Adam/Adam/ReadVariableOpReadVariableOptraining/Adam/iter'^training/Adam/Adam/AssignAddVariableOp8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp
L
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 
E
AssignVariableOpAssignVariableOptotalPlaceholder*
dtype0
_
ReadVariableOpReadVariableOptotal^AssignVariableOp*
dtype0*
_output_shapes
: 
N
Placeholder_1Placeholder*
dtype0*
_output_shapes
: *
shape: 
I
AssignVariableOp_1AssignVariableOpcountPlaceholder_1*
dtype0
c
ReadVariableOp_1ReadVariableOpcount^AssignVariableOp_1*
dtype0*
_output_shapes
: 
T
VarIsInitializedOpVarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
S
VarIsInitializedOp_1VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_2VarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpdense/kernel*
_output_shapes
: 
N
VarIsInitializedOp_4VarIsInitializedOp
dense/bias*
_output_shapes
: 
I
VarIsInitializedOp_5VarIsInitializedOpcount*
_output_shapes
: 
W
VarIsInitializedOp_6VarIsInitializedOptraining/Adam/decay*
_output_shapes
: 
a
VarIsInitializedOp_7VarIsInitializedOptraining/Adam/conv2d/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_8VarIsInitializedOptraining/Adam/conv2d_1/bias/v*
_output_shapes
: 
`
VarIsInitializedOp_9VarIsInitializedOptraining/Adam/dense/kernel/v*
_output_shapes
: 
b
VarIsInitializedOp_10VarIsInitializedOptraining/Adam/conv2d/kernel/m*
_output_shapes
: 
Y
VarIsInitializedOp_11VarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
T
VarIsInitializedOp_12VarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
P
VarIsInitializedOp_13VarIsInitializedOpconv2d/bias*
_output_shapes
: 
J
VarIsInitializedOp_14VarIsInitializedOptotal*
_output_shapes
: 
d
VarIsInitializedOp_15VarIsInitializedOptraining/Adam/conv2d_1/kernel/m*
_output_shapes
: 
a
VarIsInitializedOp_16VarIsInitializedOptraining/Adam/dense/kernel/m*
_output_shapes
: 
_
VarIsInitializedOp_17VarIsInitializedOptraining/Adam/dense/bias/m*
_output_shapes
: 
d
VarIsInitializedOp_18VarIsInitializedOptraining/Adam/conv2d_2/kernel/v*
_output_shapes
: 
_
VarIsInitializedOp_19VarIsInitializedOptraining/Adam/dense/bias/v*
_output_shapes
: 
b
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/conv2d_1/bias/m*
_output_shapes
: 
R
VarIsInitializedOp_21VarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
d
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/conv2d_2/kernel/m*
_output_shapes
: 
`
VarIsInitializedOp_23VarIsInitializedOptraining/Adam/conv2d/bias/v*
_output_shapes
: 
d
VarIsInitializedOp_24VarIsInitializedOptraining/Adam/conv2d_1/kernel/v*
_output_shapes
: 
b
VarIsInitializedOp_25VarIsInitializedOptraining/Adam/conv2d_2/bias/v*
_output_shapes
: 
Y
VarIsInitializedOp_26VarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
`
VarIsInitializedOp_27VarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
`
VarIsInitializedOp_28VarIsInitializedOptraining/Adam/conv2d/bias/m*
_output_shapes
: 
R
VarIsInitializedOp_29VarIsInitializedOpconv2d/kernel*
_output_shapes
: 
b
VarIsInitializedOp_30VarIsInitializedOptraining/Adam/conv2d_2/bias/m*
_output_shapes
: 
�
initNoOp^conv2d/bias/Assign^conv2d/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^count/Assign^dense/bias/Assign^dense/kernel/Assign^total/Assign^training/Adam/beta_1/Assign^training/Adam/beta_2/Assign#^training/Adam/conv2d/bias/m/Assign#^training/Adam/conv2d/bias/v/Assign%^training/Adam/conv2d/kernel/m/Assign%^training/Adam/conv2d/kernel/v/Assign%^training/Adam/conv2d_1/bias/m/Assign%^training/Adam/conv2d_1/bias/v/Assign'^training/Adam/conv2d_1/kernel/m/Assign'^training/Adam/conv2d_1/kernel/v/Assign%^training/Adam/conv2d_2/bias/m/Assign%^training/Adam/conv2d_2/bias/v/Assign'^training/Adam/conv2d_2/kernel/m/Assign'^training/Adam/conv2d_2/kernel/v/Assign^training/Adam/decay/Assign"^training/Adam/dense/bias/m/Assign"^training/Adam/dense/bias/v/Assign$^training/Adam/dense/kernel/m/Assign$^training/Adam/dense/kernel/v/Assign^training/Adam/iter/Assign#^training/Adam/learning_rate/Assign
(
evaluation/group_depsNoOp	^loss/mul
�
conv2d_3_inputPlaceholder*$
shape:���������FF*
dtype0*/
_output_shapes
:���������FF
�
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
:
�
.conv2d_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�hϽ*"
_class
loc:@conv2d_3/kernel
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�h�=*"
_class
loc:@conv2d_3/kernel
�
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 
�
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:@
�
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*&
_output_shapes
:@*
T0*"
_class
loc:@conv2d_3/kernel
�
conv2d_3/kernelVarHandleOp*"
_class
loc:@conv2d_3/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: * 
shared_nameconv2d_3/kernel
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
t
conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
conv2d_3/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:@
�
conv2d_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
	container *
shape:@
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
e
conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros*
dtype0
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:@
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
conv2d_3/Conv2DConv2Dconv2d_3_inputconv2d_3/Conv2D/ReadVariableOp*/
_output_shapes
:���������DD@*
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
i
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:@
�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������DD@
e
activation_4/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:���������DD@
�
max_pooling2d_3/MaxPoolMaxPoolactivation_4/Relu*
ksize
*
paddingVALID*/
_output_shapes
:���������""@*
T0*
data_formatNHWC*
strides

�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   *"
_class
loc:@conv2d_4/kernel
�
.conv2d_4/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓�*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
�
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_4/kernel*
seed2 
�
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_4/kernel
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@
�
conv2d_4/kernelVarHandleOp*
shape:@@*
dtype0*
_output_shapes
: * 
shared_nameconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
	container 
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
t
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_4/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:@
�
conv2d_4/biasVarHandleOp*
shared_nameconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
k
.conv2d_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
e
conv2d_4/bias/AssignAssignVariableOpconv2d_4/biasconv2d_4/bias/Initializer/zeros*
dtype0
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:@
g
conv2d_4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_4/Conv2DConv2Dmax_pooling2d_3/MaxPoolconv2d_4/Conv2D/ReadVariableOp*
paddingVALID*/
_output_shapes
:���������  @*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
i
conv2d_4/BiasAdd/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:@
�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:���������  @*
T0
e
activation_5/ReluReluconv2d_4/BiasAdd*
T0*/
_output_shapes
:���������  @
�
max_pooling2d_4/MaxPoolMaxPoolactivation_5/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������@
�
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
:
�
.conv2d_5/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:͓�*"
_class
loc:@conv2d_5/kernel
�
.conv2d_5/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:͓=*"
_class
loc:@conv2d_5/kernel
�
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_5/kernel*
seed2 
�
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
�
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@
�
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@
�
conv2d_5/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
	container *
shape:@@
o
0conv2d_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 
t
conv2d_5/kernel/AssignAssignVariableOpconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_5/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    * 
_class
loc:@conv2d_5/bias
�
conv2d_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
	container *
shape:@
k
.conv2d_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/bias*
_output_shapes
: 
e
conv2d_5/bias/AssignAssignVariableOpconv2d_5/biasconv2d_5/bias/Initializer/zeros*
dtype0
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
dtype0*
_output_shapes
:@
g
conv2d_5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_5/Conv2D/ReadVariableOpReadVariableOpconv2d_5/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_5/Conv2DConv2Dmax_pooling2d_4/MaxPoolconv2d_5/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������@*
	dilations

i
conv2d_5/BiasAdd/ReadVariableOpReadVariableOpconv2d_5/bias*
dtype0*
_output_shapes
:@
�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2Dconv2d_5/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:���������@*
T0
e
activation_6/ReluReluconv2d_5/BiasAdd*
T0*/
_output_shapes
:���������@
�
max_pooling2d_5/MaxPoolMaxPoolactivation_6/Relu*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
f
flatten_1/ShapeShapemax_pooling2d_5/MaxPool*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
flatten_1/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
flatten_1/Reshape/shapePackflatten_1/strided_sliceflatten_1/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
flatten_1/ReshapeReshapemax_pooling2d_5/MaxPoolflatten_1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"@     *!
_class
loc:@dense_1/kernel
�
-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�3�*!
_class
loc:@dense_1/kernel
�
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *�3=*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
�
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	�*

seed 
�
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
�
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
�
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape:	�
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	�
�
dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@dense_1/bias
�
dense_1/biasVarHandleOp*
_class
loc:@dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
m
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/MatMul/ReadVariableOp*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:���������
b
activation_7/SoftmaxSoftmaxdense_1/BiasAdd*
T0*'
_output_shapes
:���������
�
activation_7_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
z
total_1/Initializer/zerosConst*
valueB
 *    *
_class
loc:@total_1*
dtype0*
_output_shapes
: 
�
total_1VarHandleOp*
dtype0*
_output_shapes
: *
shared_name	total_1*
_class
loc:@total_1*
	container *
shape: 
_
(total_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
S
total_1/AssignAssignVariableOptotal_1total_1/Initializer/zeros*
dtype0
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
z
count_1/Initializer/zerosConst*
valueB
 *    *
_class
loc:@count_1*
dtype0*
_output_shapes
: 
�
count_1VarHandleOp*
dtype0*
_output_shapes
: *
shared_name	count_1*
_class
loc:@count_1*
	container *
shape: 
_
(count_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_1*
_output_shapes
: 
S
count_1/AssignAssignVariableOpcount_1count_1/Initializer/zeros*
dtype0
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
dtype0*
_output_shapes
: 
�
metrics_2/acc/SqueezeSqueezeactivation_7_target*#
_output_shapes
:���������*
squeeze_dims

���������*
T0
i
metrics_2/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics_2/acc/ArgMaxArgMaxactivation_7/Softmaxmetrics_2/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
}
metrics_2/acc/CastCastmetrics_2/acc/ArgMax*

SrcT0	*
Truncate( *#
_output_shapes
:���������*

DstT0
�
metrics_2/acc/EqualEqualmetrics_2/acc/Squeezemetrics_2/acc/Cast*#
_output_shapes
:���������*
incompatible_shape_error(*
T0
~
metrics_2/acc/Cast_1Castmetrics_2/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
]
metrics_2/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
metrics_2/acc/SumSummetrics_2/acc/Cast_1metrics_2/acc/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
a
!metrics_2/acc/AssignAddVariableOpAssignAddVariableOptotal_1metrics_2/acc/Sum*
dtype0
�
metrics_2/acc/ReadVariableOpReadVariableOptotal_1"^metrics_2/acc/AssignAddVariableOp^metrics_2/acc/Sum*
dtype0*
_output_shapes
: 
a
metrics_2/acc/SizeSizemetrics_2/acc/Cast_1*
T0*
out_type0*
_output_shapes
: 
p
metrics_2/acc/Cast_2Castmetrics_2/acc/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
#metrics_2/acc/AssignAddVariableOp_1AssignAddVariableOpcount_1metrics_2/acc/Cast_2"^metrics_2/acc/AssignAddVariableOp*
dtype0
�
metrics_2/acc/ReadVariableOp_1ReadVariableOpcount_1"^metrics_2/acc/AssignAddVariableOp$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics_2/acc/div_no_nan/ReadVariableOpReadVariableOptotal_1$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
)metrics_2/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount_1$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics_2/acc/div_no_nanDivNoNan'metrics_2/acc/div_no_nan/ReadVariableOp)metrics_2/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
]
metrics_2/acc/IdentityIdentitymetrics_2/acc/div_no_nan*
T0*
_output_shapes
: 
�
loss_1/activation_7_loss/CastCastactivation_7_target*
Truncate( *0
_output_shapes
:������������������*

DstT0	*

SrcT0
m
loss_1/activation_7_loss/ShapeShapedense_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
y
&loss_1/activation_7_loss/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
 loss_1/activation_7_loss/ReshapeReshapeloss_1/activation_7_loss/Cast&loss_1/activation_7_loss/Reshape/shape*#
_output_shapes
:���������*
T0	*
Tshape0

,loss_1/activation_7_loss/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������
x
.loss_1/activation_7_loss/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
x
.loss_1/activation_7_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
&loss_1/activation_7_loss/strided_sliceStridedSliceloss_1/activation_7_loss/Shape,loss_1/activation_7_loss/strided_slice/stack.loss_1/activation_7_loss/strided_slice/stack_1.loss_1/activation_7_loss/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
u
*loss_1/activation_7_loss/Reshape_1/shape/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
(loss_1/activation_7_loss/Reshape_1/shapePack*loss_1/activation_7_loss/Reshape_1/shape/0&loss_1/activation_7_loss/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
"loss_1/activation_7_loss/Reshape_1Reshapedense_1/BiasAdd(loss_1/activation_7_loss/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
Bloss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShape loss_1/activation_7_loss/Reshape*
T0	*
out_type0*
_output_shapes
:
�
`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits"loss_1/activation_7_loss/Reshape_1 loss_1/activation_7_loss/Reshape*?
_output_shapes-
+:���������:������������������*
Tlabels0	*
T0
r
-loss_1/activation_7_loss/weighted_loss/Cast/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
[loss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Zloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Zloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
Yloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
q
iloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Hloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsj^loss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0*
out_type0
�
Hloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/ConstConstj^loss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_likeFillHloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/ShapeHloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0*

index_type0
�
8loss_1/activation_7_loss/weighted_loss/broadcast_weightsMul-loss_1/activation_7_loss/weighted_loss/Cast/xBloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
*loss_1/activation_7_loss/weighted_loss/MulMul`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits8loss_1/activation_7_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
h
loss_1/activation_7_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss_1/activation_7_loss/SumSum*loss_1/activation_7_loss/weighted_loss/Mulloss_1/activation_7_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
%loss_1/activation_7_loss/num_elementsSize*loss_1/activation_7_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 
�
*loss_1/activation_7_loss/num_elements/CastCast%loss_1/activation_7_loss/num_elements*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
c
 loss_1/activation_7_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
�
loss_1/activation_7_loss/Sum_1Sumloss_1/activation_7_loss/Sum loss_1/activation_7_loss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
loss_1/activation_7_loss/valueDivNoNanloss_1/activation_7_loss/Sum_1*loss_1/activation_7_loss/num_elements/Cast*
_output_shapes
: *
T0
Q
loss_1/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
`

loss_1/mulMulloss_1/mul/xloss_1/activation_7_loss/value*
_output_shapes
: *
T0
l
)training_2/Adam/gradients/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
r
-training_2/Adam/gradients/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
(training_2/Adam/gradients/gradients/FillFill)training_2/Adam/gradients/gradients/Shape-training_2/Adam/gradients/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
7training_2/Adam/gradients/gradients/loss_1/mul_grad/MulMul(training_2/Adam/gradients/gradients/Fillloss_1/activation_7_loss/value*
T0*
_output_shapes
: 
�
9training_2/Adam/gradients/gradients/loss_1/mul_grad/Mul_1Mul(training_2/Adam/gradients/gradients/Fillloss_1/mul/x*
T0*
_output_shapes
: 
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
]training_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ShapeOtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Rtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nanDivNoNan9training_2/Adam/gradients/gradients/loss_1/mul_grad/Mul_1*loss_1/activation_7_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/SumSumRtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan]training_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ReshapeReshapeKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/SumMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/NegNegloss_1/activation_7_loss/Sum_1*
T0*
_output_shapes
: 
�
Ttraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_1DivNoNanKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Neg*loss_1/activation_7_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ttraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_2DivNoNanTtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_1*loss_1/activation_7_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/mulMul9training_2/Adam/gradients/gradients/loss_1/mul_grad/Mul_1Ttraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_2*
_output_shapes
: *
T0
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Sum_1SumKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/mul_training_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
Qtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Reshape_1ReshapeMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Sum_1Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Utraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/ReshapeReshapeOtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ReshapeUtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
�
Ltraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/TileTileOtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/ReshapeMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/Const*
T0*
_output_shapes
: *

Tmultiples0
�
Straining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/ReshapeReshapeLtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/TileStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/ShapeShape*loss_1/activation_7_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
�
Jtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/TileTileMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/ReshapeKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
�
Ytraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/ShapeShape`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape_1Shape8loss_1/activation_7_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
itraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Wtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/MulMulJtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Tile8loss_1/activation_7_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
�
Wtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/SumSumWtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Mulitraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/ReshapeReshapeWtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/SumYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
Ytraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Mul_1Mul`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsJtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
Ytraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Sum_1SumYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Mul_1ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
]training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Reshape_1ReshapeYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Sum_1[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
.training_2/Adam/gradients/gradients/zeros_like	ZerosLikebloss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientbloss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Reshape�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*0
_output_shapes
:������������������*
T0
�
Qtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/ShapeShapedense_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Straining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/ReshapeReshape�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulQtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Dtraining_2/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
�
>training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMulMatMulStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Reshapedense_1/MatMul/ReadVariableOp*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
@training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/ReshapeStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Reshape*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
@training_2/Adam/gradients/gradients/flatten_1/Reshape_grad/ShapeShapemax_pooling2d_5/MaxPool*
_output_shapes
:*
T0*
out_type0
�
Btraining_2/Adam/gradients/gradients/flatten_1/Reshape_grad/ReshapeReshape>training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul@training_2/Adam/gradients/gradients/flatten_1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
Ltraining_2/Adam/gradients/gradients/max_pooling2d_5/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_6/Relumax_pooling2d_5/MaxPoolBtraining_2/Adam/gradients/gradients/flatten_1/Reshape_grad/Reshape*
ksize
*
paddingVALID*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

�
Ctraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGradReluGradLtraining_2/Adam/gradients/gradients/max_pooling2d_5/MaxPool_grad/MaxPoolGradactivation_6/Relu*/
_output_shapes
:���������@*
T0
�
Etraining_2/Adam/gradients/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGradCtraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
?training_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/ShapeNShapeNmax_pooling2d_4/MaxPoolconv2d_5/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Ltraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_5/Conv2D/ReadVariableOpCtraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGrad*
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
:���������@
�
Mtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_4/MaxPoolAtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/ShapeN:1Ctraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@@*
	dilations
*
T0
�
Ltraining_2/Adam/gradients/gradients/max_pooling2d_4/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_5/Relumax_pooling2d_4/MaxPoolLtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������  @*
T0
�
Ctraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGradReluGradLtraining_2/Adam/gradients/gradients/max_pooling2d_4/MaxPool_grad/MaxPoolGradactivation_5/Relu*
T0*/
_output_shapes
:���������  @
�
Etraining_2/Adam/gradients/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGradCtraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
?training_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/ShapeNShapeNmax_pooling2d_3/MaxPoolconv2d_4/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Ltraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_4/Conv2D/ReadVariableOpCtraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������""@*
	dilations

�
Mtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_3/MaxPoolAtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/ShapeN:1Ctraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGrad*&
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
�
Ltraining_2/Adam/gradients/gradients/max_pooling2d_3/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_4/Relumax_pooling2d_3/MaxPoolLtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������DD@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
Ctraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGradReluGradLtraining_2/Adam/gradients/gradients/max_pooling2d_3/MaxPool_grad/MaxPoolGradactivation_4/Relu*
T0*/
_output_shapes
:���������DD@
�
Etraining_2/Adam/gradients/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradCtraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
?training_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/ShapeNShapeNconv2d_3_inputconv2d_3/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Ltraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_3/Conv2D/ReadVariableOpCtraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGrad*/
_output_shapes
:���������FF*
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
�
Mtraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_3_inputAtraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/ShapeN:1Ctraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGrad*
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
:@
�
&training_2/Adam/iter/Initializer/zerosConst*
value	B	 R *'
_class
loc:@training_2/Adam/iter*
dtype0	*
_output_shapes
: 
�
training_2/Adam/iterVarHandleOp*%
shared_nametraining_2/Adam/iter*'
_class
loc:@training_2/Adam/iter*
	container *
shape: *
dtype0	*
_output_shapes
: 
y
5training_2/Adam/iter/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/iter*
_output_shapes
: 
z
training_2/Adam/iter/AssignAssignVariableOptraining_2/Adam/iter&training_2/Adam/iter/Initializer/zeros*
dtype0	
u
(training_2/Adam/iter/Read/ReadVariableOpReadVariableOptraining_2/Adam/iter*
dtype0	*
_output_shapes
: 
�
0training_2/Adam/beta_1/Initializer/initial_valueConst*
valueB
 *fff?*)
_class
loc:@training_2/Adam/beta_1*
dtype0*
_output_shapes
: 
�
training_2/Adam/beta_1VarHandleOp*'
shared_nametraining_2/Adam/beta_1*)
_class
loc:@training_2/Adam/beta_1*
	container *
shape: *
dtype0*
_output_shapes
: 
}
7training_2/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/beta_1*
_output_shapes
: 
�
training_2/Adam/beta_1/AssignAssignVariableOptraining_2/Adam/beta_10training_2/Adam/beta_1/Initializer/initial_value*
dtype0
y
*training_2/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
dtype0*
_output_shapes
: 
�
0training_2/Adam/beta_2/Initializer/initial_valueConst*
valueB
 *w�?*)
_class
loc:@training_2/Adam/beta_2*
dtype0*
_output_shapes
: 
�
training_2/Adam/beta_2VarHandleOp*
shape: *
dtype0*
_output_shapes
: *'
shared_nametraining_2/Adam/beta_2*)
_class
loc:@training_2/Adam/beta_2*
	container 
}
7training_2/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/beta_2*
_output_shapes
: 
�
training_2/Adam/beta_2/AssignAssignVariableOptraining_2/Adam/beta_20training_2/Adam/beta_2/Initializer/initial_value*
dtype0
y
*training_2/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
dtype0*
_output_shapes
: 
�
/training_2/Adam/decay/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    *(
_class
loc:@training_2/Adam/decay
�
training_2/Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *&
shared_nametraining_2/Adam/decay*(
_class
loc:@training_2/Adam/decay*
	container *
shape: 
{
6training_2/Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/decay*
_output_shapes
: 
�
training_2/Adam/decay/AssignAssignVariableOptraining_2/Adam/decay/training_2/Adam/decay/Initializer/initial_value*
dtype0
w
)training_2/Adam/decay/Read/ReadVariableOpReadVariableOptraining_2/Adam/decay*
dtype0*
_output_shapes
: 
�
7training_2/Adam/learning_rate/Initializer/initial_valueConst*
valueB
 *o�:*0
_class&
$"loc:@training_2/Adam/learning_rate*
dtype0*
_output_shapes
: 
�
training_2/Adam/learning_rateVarHandleOp*.
shared_nametraining_2/Adam/learning_rate*0
_class&
$"loc:@training_2/Adam/learning_rate*
	container *
shape: *
dtype0*
_output_shapes
: 
�
>training_2/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/learning_rate*
_output_shapes
: 
�
$training_2/Adam/learning_rate/AssignAssignVariableOptraining_2/Adam/learning_rate7training_2/Adam/learning_rate/Initializer/initial_value*
dtype0
�
1training_2/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_3/kernel/m/Initializer/zerosConst*"
_class
loc:@conv2d_3/kernel*%
valueB@*    *
dtype0*&
_output_shapes
:@
�
!training_2/Adam/conv2d_3/kernel/mVarHandleOp*2
shared_name#!training_2/Adam/conv2d_3/kernel/m*"
_class
loc:@conv2d_3/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
Btraining_2/Adam/conv2d_3/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_3/kernel/m*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel
�
(training_2/Adam/conv2d_3/kernel/m/AssignAssignVariableOp!training_2/Adam/conv2d_3/kernel/m3training_2/Adam/conv2d_3/kernel/m/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_3/kernel/m*"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
1training_2/Adam/conv2d_3/bias/m/Initializer/zerosConst*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_3/bias*
valueB@*    
�
training_2/Adam/conv2d_3/bias/mVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_3/bias/m* 
_class
loc:@conv2d_3/bias*
	container *
shape:@
�
@training_2/Adam/conv2d_3/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_3/bias/m* 
_class
loc:@conv2d_3/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_3/bias/m/AssignAssignVariableOptraining_2/Adam/conv2d_3/bias/m1training_2/Adam/conv2d_3/bias/m/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_3/bias/m*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_3/bias
�
Ctraining_2/Adam/conv2d_4/kernel/m/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_4/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
9training_2/Adam/conv2d_4/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_4/kernel/m/Initializer/zerosFillCtraining_2/Adam/conv2d_4/kernel/m/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_4/kernel/m/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_4/kernel*

index_type0*&
_output_shapes
:@@
�
!training_2/Adam/conv2d_4/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_4/kernel/m*"
_class
loc:@conv2d_4/kernel*
	container *
shape:@@
�
Btraining_2/Adam/conv2d_4/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_4/kernel/m*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_4/kernel/m/AssignAssignVariableOp!training_2/Adam/conv2d_4/kernel/m3training_2/Adam/conv2d_4/kernel/m/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_4/kernel/m*
dtype0*&
_output_shapes
:@@*"
_class
loc:@conv2d_4/kernel
�
1training_2/Adam/conv2d_4/bias/m/Initializer/zerosConst* 
_class
loc:@conv2d_4/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_4/bias/mVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_4/bias/m* 
_class
loc:@conv2d_4/bias*
	container 
�
@training_2/Adam/conv2d_4/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_4/bias/m* 
_class
loc:@conv2d_4/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_4/bias/m/AssignAssignVariableOptraining_2/Adam/conv2d_4/bias/m1training_2/Adam/conv2d_4/bias/m/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_4/bias/m* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:@
�
Ctraining_2/Adam/conv2d_5/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_5/kernel*%
valueB"      @   @   
�
9training_2/Adam/conv2d_5/kernel/m/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_5/kernel/m/Initializer/zerosFillCtraining_2/Adam/conv2d_5/kernel/m/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_5/kernel/m/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_5/kernel*

index_type0*&
_output_shapes
:@@
�
!training_2/Adam/conv2d_5/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_5/kernel/m*"
_class
loc:@conv2d_5/kernel*
	container *
shape:@@
�
Btraining_2/Adam/conv2d_5/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_5/kernel/m*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_5/kernel/m/AssignAssignVariableOp!training_2/Adam/conv2d_5/kernel/m3training_2/Adam/conv2d_5/kernel/m/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_5/kernel/m*"
_class
loc:@conv2d_5/kernel*
dtype0*&
_output_shapes
:@@
�
1training_2/Adam/conv2d_5/bias/m/Initializer/zerosConst* 
_class
loc:@conv2d_5/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_5/bias/mVarHandleOp*0
shared_name!training_2/Adam/conv2d_5/bias/m* 
_class
loc:@conv2d_5/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
@training_2/Adam/conv2d_5/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_5/bias/m*
_output_shapes
: * 
_class
loc:@conv2d_5/bias
�
&training_2/Adam/conv2d_5/bias/m/AssignAssignVariableOptraining_2/Adam/conv2d_5/bias/m1training_2/Adam/conv2d_5/bias/m/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_5/bias/m*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_5/bias
�
Btraining_2/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_1/kernel*
valueB"@     
�
8training_2/Adam/dense_1/kernel/m/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
2training_2/Adam/dense_1/kernel/m/Initializer/zerosFillBtraining_2/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensor8training_2/Adam/dense_1/kernel/m/Initializer/zeros/Const*
_output_shapes
:	�*
T0*!
_class
loc:@dense_1/kernel*

index_type0
�
 training_2/Adam/dense_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *1
shared_name" training_2/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
	container *
shape:	�
�
Atraining_2/Adam/dense_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp training_2/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
�
'training_2/Adam/dense_1/kernel/m/AssignAssignVariableOp training_2/Adam/dense_1/kernel/m2training_2/Adam/dense_1/kernel/m/Initializer/zeros*
dtype0
�
4training_2/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:	�
�
0training_2/Adam/dense_1/bias/m/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
training_2/Adam/dense_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: */
shared_name training_2/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
	container *
shape:
�
?training_2/Adam/dense_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/dense_1/bias/m*
_output_shapes
: *
_class
loc:@dense_1/bias
�
%training_2/Adam/dense_1/bias/m/AssignAssignVariableOptraining_2/Adam/dense_1/bias/m0training_2/Adam/dense_1/bias/m/Initializer/zeros*
dtype0
�
2training_2/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
�
3training_2/Adam/conv2d_3/kernel/v/Initializer/zerosConst*"
_class
loc:@conv2d_3/kernel*%
valueB@*    *
dtype0*&
_output_shapes
:@
�
!training_2/Adam/conv2d_3/kernel/vVarHandleOp*2
shared_name#!training_2/Adam/conv2d_3/kernel/v*"
_class
loc:@conv2d_3/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
Btraining_2/Adam/conv2d_3/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_3/kernel/v*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel
�
(training_2/Adam/conv2d_3/kernel/v/AssignAssignVariableOp!training_2/Adam/conv2d_3/kernel/v3training_2/Adam/conv2d_3/kernel/v/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_3/kernel/v*"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
1training_2/Adam/conv2d_3/bias/v/Initializer/zerosConst* 
_class
loc:@conv2d_3/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_3/bias/vVarHandleOp*0
shared_name!training_2/Adam/conv2d_3/bias/v* 
_class
loc:@conv2d_3/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
@training_2/Adam/conv2d_3/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_3/bias/v* 
_class
loc:@conv2d_3/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_3/bias/v/AssignAssignVariableOptraining_2/Adam/conv2d_3/bias/v1training_2/Adam/conv2d_3/bias/v/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_3/bias/v* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:@
�
Ctraining_2/Adam/conv2d_4/kernel/v/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_4/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
9training_2/Adam/conv2d_4/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_4/kernel/v/Initializer/zerosFillCtraining_2/Adam/conv2d_4/kernel/v/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_4/kernel/v/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_4/kernel*

index_type0*&
_output_shapes
:@@
�
!training_2/Adam/conv2d_4/kernel/vVarHandleOp*"
_class
loc:@conv2d_4/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_4/kernel/v
�
Btraining_2/Adam/conv2d_4/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_4/kernel/v*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_4/kernel/v/AssignAssignVariableOp!training_2/Adam/conv2d_4/kernel/v3training_2/Adam/conv2d_4/kernel/v/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_4/kernel/v*"
_class
loc:@conv2d_4/kernel*
dtype0*&
_output_shapes
:@@
�
1training_2/Adam/conv2d_4/bias/v/Initializer/zerosConst*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_4/bias*
valueB@*    
�
training_2/Adam/conv2d_4/bias/vVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_4/bias/v* 
_class
loc:@conv2d_4/bias*
	container *
shape:@
�
@training_2/Adam/conv2d_4/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_4/bias/v* 
_class
loc:@conv2d_4/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_4/bias/v/AssignAssignVariableOptraining_2/Adam/conv2d_4/bias/v1training_2/Adam/conv2d_4/bias/v/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_4/bias/v* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:@
�
Ctraining_2/Adam/conv2d_5/kernel/v/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@conv2d_5/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
9training_2/Adam/conv2d_5/kernel/v/Initializer/zeros/ConstConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_5/kernel/v/Initializer/zerosFillCtraining_2/Adam/conv2d_5/kernel/v/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_5/kernel/v/Initializer/zeros/Const*
T0*"
_class
loc:@conv2d_5/kernel*

index_type0*&
_output_shapes
:@@
�
!training_2/Adam/conv2d_5/kernel/vVarHandleOp*
	container *
shape:@@*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_5/kernel/v*"
_class
loc:@conv2d_5/kernel
�
Btraining_2/Adam/conv2d_5/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_5/kernel/v*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel
�
(training_2/Adam/conv2d_5/kernel/v/AssignAssignVariableOp!training_2/Adam/conv2d_5/kernel/v3training_2/Adam/conv2d_5/kernel/v/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_5/kernel/v*"
_class
loc:@conv2d_5/kernel*
dtype0*&
_output_shapes
:@@
�
1training_2/Adam/conv2d_5/bias/v/Initializer/zerosConst* 
_class
loc:@conv2d_5/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_5/bias/vVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_5/bias/v* 
_class
loc:@conv2d_5/bias*
	container *
shape:@
�
@training_2/Adam/conv2d_5/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_5/bias/v*
_output_shapes
: * 
_class
loc:@conv2d_5/bias
�
&training_2/Adam/conv2d_5/bias/v/AssignAssignVariableOptraining_2/Adam/conv2d_5/bias/v1training_2/Adam/conv2d_5/bias/v/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_5/bias/v* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
:@
�
Btraining_2/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
valueB"@     *
dtype0*
_output_shapes
:
�
8training_2/Adam/dense_1/kernel/v/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
2training_2/Adam/dense_1/kernel/v/Initializer/zerosFillBtraining_2/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensor8training_2/Adam/dense_1/kernel/v/Initializer/zeros/Const*
_output_shapes
:	�*
T0*!
_class
loc:@dense_1/kernel*

index_type0
�
 training_2/Adam/dense_1/kernel/vVarHandleOp*1
shared_name" training_2/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
: 
�
Atraining_2/Adam/dense_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp training_2/Adam/dense_1/kernel/v*
_output_shapes
: *!
_class
loc:@dense_1/kernel
�
'training_2/Adam/dense_1/kernel/v/AssignAssignVariableOp training_2/Adam/dense_1/kernel/v2training_2/Adam/dense_1/kernel/v/Initializer/zeros*
dtype0
�
4training_2/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_1/kernel/v*
dtype0*
_output_shapes
:	�*!
_class
loc:@dense_1/kernel
�
0training_2/Adam/dense_1/bias/v/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
training_2/Adam/dense_1/bias/vVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: */
shared_name training_2/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias
�
?training_2/Adam/dense_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
_output_shapes
: 
�
%training_2/Adam/dense_1/bias/v/AssignAssignVariableOptraining_2/Adam/dense_1/bias/v0training_2/Adam/dense_1/bias/v/Initializer/zeros*
dtype0
�
2training_2/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
}
'training_2/Adam/Identity/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
dtype0*
_output_shapes
: 
n
training_2/Adam/IdentityIdentity'training_2/Adam/Identity/ReadVariableOp*
_output_shapes
: *
T0
k
training_2/Adam/ReadVariableOpReadVariableOptraining_2/Adam/iter*
dtype0	*
_output_shapes
: 
W
training_2/Adam/add/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
t
training_2/Adam/addAddV2training_2/Adam/ReadVariableOptraining_2/Adam/add/y*
_output_shapes
: *
T0	
q
training_2/Adam/CastCasttraining_2/Adam/add*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
x
)training_2/Adam/Identity_1/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
dtype0*
_output_shapes
: 
r
training_2/Adam/Identity_1Identity)training_2/Adam/Identity_1/ReadVariableOp*
T0*
_output_shapes
: 
x
)training_2/Adam/Identity_2/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
dtype0*
_output_shapes
: 
r
training_2/Adam/Identity_2Identity)training_2/Adam/Identity_2/ReadVariableOp*
T0*
_output_shapes
: 
m
training_2/Adam/PowPowtraining_2/Adam/Identity_1training_2/Adam/Cast*
T0*
_output_shapes
: 
o
training_2/Adam/Pow_1Powtraining_2/Adam/Identity_2training_2/Adam/Cast*
T0*
_output_shapes
: 
Z
training_2/Adam/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
i
training_2/Adam/subSubtraining_2/Adam/sub/xtraining_2/Adam/Pow_1*
T0*
_output_shapes
: 
R
training_2/Adam/SqrtSqrttraining_2/Adam/sub*
_output_shapes
: *
T0
\
training_2/Adam/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
training_2/Adam/sub_1Subtraining_2/Adam/sub_1/xtraining_2/Adam/Pow*
_output_shapes
: *
T0
p
training_2/Adam/truedivRealDivtraining_2/Adam/Sqrttraining_2/Adam/sub_1*
T0*
_output_shapes
: 
n
training_2/Adam/mulMultraining_2/Adam/Identitytraining_2/Adam/truediv*
T0*
_output_shapes
: 
Z
training_2/Adam/ConstConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
\
training_2/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
r
training_2/Adam/sub_2Subtraining_2/Adam/sub_2/xtraining_2/Adam/Identity_1*
T0*
_output_shapes
: 
\
training_2/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
r
training_2/Adam/sub_3Subtraining_2/Adam/sub_3/xtraining_2/Adam/Identity_2*
_output_shapes
: *
T0
�
=training_2/Adam/Adam/update_conv2d_3/kernel/ResourceApplyAdamResourceApplyAdamconv2d_3/kernel!training_2/Adam/conv2d_3/kernel/m!training_2/Adam/conv2d_3/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstMtraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*
use_nesterov( *
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel
�
;training_2/Adam/Adam/update_conv2d_3/bias/ResourceApplyAdamResourceApplyAdamconv2d_3/biastraining_2/Adam/conv2d_3/bias/mtraining_2/Adam/conv2d_3/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstEtraining_2/Adam/gradients/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
use_nesterov( 
�
=training_2/Adam/Adam/update_conv2d_4/kernel/ResourceApplyAdamResourceApplyAdamconv2d_4/kernel!training_2/Adam/conv2d_4/kernel/m!training_2/Adam/conv2d_4/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstMtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*
use_nesterov( *
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel
�
;training_2/Adam/Adam/update_conv2d_4/bias/ResourceApplyAdamResourceApplyAdamconv2d_4/biastraining_2/Adam/conv2d_4/bias/mtraining_2/Adam/conv2d_4/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstEtraining_2/Adam/gradients/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
use_nesterov( 
�
=training_2/Adam/Adam/update_conv2d_5/kernel/ResourceApplyAdamResourceApplyAdamconv2d_5/kernel!training_2/Adam/conv2d_5/kernel/m!training_2/Adam/conv2d_5/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstMtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter*
T0*"
_class
loc:@conv2d_5/kernel*
use_nesterov( *
use_locking(
�
;training_2/Adam/Adam/update_conv2d_5/bias/ResourceApplyAdamResourceApplyAdamconv2d_5/biastraining_2/Adam/conv2d_5/bias/mtraining_2/Adam/conv2d_5/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstEtraining_2/Adam/gradients/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
use_nesterov( 
�
<training_2/Adam/Adam/update_dense_1/kernel/ResourceApplyAdamResourceApplyAdamdense_1/kernel training_2/Adam/dense_1/kernel/m training_2/Adam/dense_1/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/Const@training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
use_nesterov( 
�
:training_2/Adam/Adam/update_dense_1/bias/ResourceApplyAdamResourceApplyAdamdense_1/biastraining_2/Adam/dense_1/bias/mtraining_2/Adam/dense_1/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstDtraining_2/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0*
_class
loc:@dense_1/bias
�
training_2/Adam/Adam/ConstConst<^training_2/Adam/Adam/update_conv2d_3/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_3/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_4/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_4/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_5/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_5/kernel/ResourceApplyAdam;^training_2/Adam/Adam/update_dense_1/bias/ResourceApplyAdam=^training_2/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam*
value	B	 R*
dtype0	*
_output_shapes
: 
~
(training_2/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining_2/Adam/itertraining_2/Adam/Adam/Const*
dtype0	
�
#training_2/Adam/Adam/ReadVariableOpReadVariableOptraining_2/Adam/iter)^training_2/Adam/Adam/AssignAddVariableOp<^training_2/Adam/Adam/update_conv2d_3/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_3/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_4/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_4/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_5/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_5/kernel/ResourceApplyAdam;^training_2/Adam/Adam/update_dense_1/bias/ResourceApplyAdam=^training_2/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
U
training_3/group_depsNoOp^loss_1/mul)^training_2/Adam/Adam/AssignAddVariableOp"��
K�|     �F?	��v����AJ��	
�*�*
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
2	��
�
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
dtypetype�
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
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
�
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
�
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
incompatible_shape_errorbool(�
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
�
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
�
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
2	�
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
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
�
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
use_nesterovbool( �
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
�
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
�
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
�
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
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�
&
	ZerosLike
x"T
y"T"	
Ttype*1.15.02v1.15.0-rc3-22-g590d6ee��

conv2d_inputPlaceholder*
dtype0*/
_output_shapes
:���������FF*$
shape:���������FF
�
.conv2d/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@conv2d/kernel*%
valueB"         @   *
dtype0*
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/minConst* 
_class
loc:@conv2d/kernel*
valueB
 *�hϽ*
dtype0*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@conv2d/kernel*
valueB
 *�h�=*
dtype0*
_output_shapes
: 
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
�
conv2d/kernelVarHandleOp* 
_class
loc:@conv2d/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d/kernel
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
�
conv2d/bias/Initializer/zerosConst*
_class
loc:@conv2d/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
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
�
conv2d/Conv2DConv2Dconv2d_inputconv2d/Conv2D/ReadVariableOp*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������DD@*
	dilations

e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:@
�
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������DD@
a
activation/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:���������DD@
�
max_pooling2d/MaxPoolMaxPoolactivation/Relu*/
_output_shapes
:���������""@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_1/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *:͓�*
dtype0*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
valueB
 *:͓=
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
�
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@
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
�
conv2d_1/bias/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
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
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*/
_output_shapes
:���������  @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@
�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:���������  @*
T0
e
activation_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:���������  @
�
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu*
ksize
*
paddingVALID*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

�
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_2/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
.conv2d_2/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *:͓�*
dtype0*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *:͓=*
dtype0*
_output_shapes
: 
�
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:@@*

seed 
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_2/kernel
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_2/kernel
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
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
�
conv2d_2/bias/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_2/biasVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
	container 
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
�
conv2d_2/Conv2DConv2Dmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
paddingVALID*/
_output_shapes
:���������@*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@
�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
�
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
d
flatten/ShapeShapemax_pooling2d_2/MaxPool*
_output_shapes
:*
T0*
out_type0
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
�
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
b
flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
�
flatten/ReshapeReshapemax_pooling2d_2/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"@     *
dtype0*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *�3�
�
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *�3=*
dtype0*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�*
T0*
_class
loc:@dense/kernel
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
�
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense/kernel*
_class
loc:@dense/kernel*
	container *
shape:	�
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
:	�
�
dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/bias*
valueB*    
�

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
:	�
�
dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
�
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:���������
`
activation_3/SoftmaxSoftmaxdense/BiasAdd*
T0*'
_output_shapes
:���������
�
activation_3_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
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
�
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
count/Initializer/zerosConst*
_class

loc:@count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
countVarHandleOp*
dtype0*
_output_shapes
: *
shared_namecount*
_class

loc:@count*
	container *
shape: 
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
�
metrics/acc/SqueezeSqueezeactivation_3_target*
squeeze_dims

���������*
T0*#
_output_shapes
:���������
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxactivation_3/Softmaxmetrics/acc/ArgMax/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
y
metrics/acc/CastCastmetrics/acc/ArgMax*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:���������
�
metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast*#
_output_shapes
:���������*
incompatible_shape_error(*
T0
z
metrics/acc/Cast_1Castmetrics/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
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
: *

Tidx0*
	keep_dims( 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
�
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
�
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_2 ^metrics/acc/AssignAddVariableOp*
dtype0
�
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
_output_shapes
: *
T0
�
loss/activation_3_loss/CastCastactivation_3_target*

SrcT0*
Truncate( *

DstT0	*0
_output_shapes
:������������������
i
loss/activation_3_loss/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
$loss/activation_3_loss/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
loss/activation_3_loss/ReshapeReshapeloss/activation_3_loss/Cast$loss/activation_3_loss/Reshape/shape*#
_output_shapes
:���������*
T0	*
Tshape0
}
*loss/activation_3_loss/strided_slice/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:
v
,loss/activation_3_loss/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,loss/activation_3_loss/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
$loss/activation_3_loss/strided_sliceStridedSliceloss/activation_3_loss/Shape*loss/activation_3_loss/strided_slice/stack,loss/activation_3_loss/strided_slice/stack_1,loss/activation_3_loss/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
s
(loss/activation_3_loss/Reshape_1/shape/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
&loss/activation_3_loss/Reshape_1/shapePack(loss/activation_3_loss/Reshape_1/shape/0$loss/activation_3_loss/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
 loss/activation_3_loss/Reshape_1Reshapedense/BiasAdd&loss/activation_3_loss/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
@loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/activation_3_loss/Reshape*
_output_shapes
:*
T0	*
out_type0
�
^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits loss/activation_3_loss/Reshape_1loss/activation_3_loss/Reshape*
T0*
Tlabels0	*?
_output_shapes-
+:���������:������������������
p
+loss/activation_3_loss/weighted_loss/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
Yloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
�
Xloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
Wloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gloss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsh^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0*
out_type0
�
Floss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ConstConsth^loss/activation_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_likeFillFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeFloss/activation_3_loss/weighted_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0*

index_type0
�
6loss/activation_3_loss/weighted_loss/broadcast_weightsMul+loss/activation_3_loss/weighted_loss/Cast/x@loss/activation_3_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
(loss/activation_3_loss/weighted_loss/MulMul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
f
loss/activation_3_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
loss/activation_3_loss/SumSum(loss/activation_3_loss/weighted_loss/Mulloss/activation_3_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
#loss/activation_3_loss/num_elementsSize(loss/activation_3_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 
�
(loss/activation_3_loss/num_elements/CastCast#loss/activation_3_loss/num_elements*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
a
loss/activation_3_loss/Const_1Const*
dtype0*
_output_shapes
: *
valueB 
�
loss/activation_3_loss/Sum_1Sumloss/activation_3_loss/Sumloss/activation_3_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss/activation_3_loss/valueDivNoNanloss/activation_3_loss/Sum_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Z
loss/mulMul
loss/mul/xloss/activation_3_loss/value*
_output_shapes
: *
T0
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
 *  �?*
dtype0*
_output_shapes
: 
�
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
�
3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/activation_3_loss/value*
_output_shapes
: *
T0
�
5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
_output_shapes
: *
T0
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ShapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ntraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumSumNtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nanYtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/SumItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/NegNegloss/activation_3_loss/Sum_1*
T0*
_output_shapes
: 
�
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1DivNoNanGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Neg(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2DivNoNanPtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_1(loss/activation_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ptraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1SumGtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/mul[training/Adam/gradients/gradients/loss/activation_3_loss/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
Mtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Reshape_1ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Sum_1Ktraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Qtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Ktraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeReshapeKtraining/Adam/gradients/gradients/loss/activation_3_loss/value_grad/ReshapeQtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
�
Htraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileTileKtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/ReshapeItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
�
Otraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
Itraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeReshapeHtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_1_grad/TileOtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
Gtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ShapeShape(loss/activation_3_loss/weighted_loss/Mul*
_output_shapes
:*
T0*
out_type0
�
Ftraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/TileTileItraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/ReshapeGtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
�
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeShape^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1Shape6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
etraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ShapeWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/MulMulFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile6loss/activation_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
�
Straining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumSumStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Muletraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/ReshapeReshapeStraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1Mul^loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsFtraining/Adam/gradients/gradients/loss/activation_3_loss/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
Utraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1SumUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Mul_1gtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ytraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshape_1ReshapeUtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Sum_1Wtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
,training/Adam/gradients/gradients/zeros_like	ZerosLike`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient`loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*0
_output_shapes
:������������������
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsWtraining/Adam/gradients/gradients/loss/activation_3_loss/weighted_loss/Mul_grad/Reshape�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*0
_output_shapes
:������������������*
T0
�
Mtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Otraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/ReshapeReshape�training/Adam/gradients/gradients/loss/activation_3_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
�
:training/Adam/gradients/gradients/dense/MatMul_grad/MatMulMatMulOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshapedense/MatMul/ReadVariableOp*
T0*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1MatMulflatten/ReshapeOtraining/Adam/gradients/gradients/loss/activation_3_loss/Reshape_1_grad/Reshape*
transpose_a(*
_output_shapes
:	�*
transpose_b( *
T0
�
<training/Adam/gradients/gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_2/MaxPool*
T0*
out_type0*
_output_shapes
:
�
>training/Adam/gradients/gradients/flatten/Reshape_grad/ReshapeReshape:training/Adam/gradients/gradients/dense/MatMul_grad/MatMul<training/Adam/gradients/gradients/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������@
�
Jtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_2/Relumax_pooling2d_2/MaxPool>training/Adam/gradients/gradients/flatten/Reshape_grad/Reshape*
ksize
*
paddingVALID*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

�
Atraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradactivation_2/Relu*
T0*/
_output_shapes
:���������@
�
Ctraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Jtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_2/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������@*
	dilations
*
T0
�
Ktraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool?training/Adam/gradients/gradients/conv2d_2/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/gradients/activation_2/Relu_grad/ReluGrad*
paddingVALID*&
_output_shapes
:@@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
�
Jtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_1/Relumax_pooling2d_1/MaxPoolJtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:���������  @*
T0*
strides
*
data_formatNHWC
�
Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGradReluGradJtraining/Adam/gradients/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradactivation_1/Relu*
T0*/
_output_shapes
:���������  @
�
Ctraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Jtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/Conv2D/ReadVariableOpAtraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
paddingVALID*/
_output_shapes
:���������""@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
�
Ktraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool?training/Adam/gradients/gradients/conv2d_1/Conv2D_grad/ShapeN:1Atraining/Adam/gradients/gradients/activation_1/Relu_grad/ReluGrad*
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
�
Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradactivation/Relumax_pooling2d/MaxPoolJtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:���������DD@*
T0*
data_formatNHWC*
strides

�
?training/Adam/gradients/gradients/activation/Relu_grad/ReluGradReluGradHtraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradactivation/Relu*/
_output_shapes
:���������DD@*
T0
�
Atraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNShapeNconv2d_inputconv2d/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Htraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput;training/Adam/gradients/gradients/conv2d/Conv2D_grad/ShapeNconv2d/Conv2D/ReadVariableOp?training/Adam/gradients/gradients/activation/Relu_grad/ReluGrad*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������FF
�
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
�
$training/Adam/iter/Initializer/zerosConst*%
_class
loc:@training/Adam/iter*
value	B	 R *
dtype0	*
_output_shapes
: 
�
training/Adam/iterVarHandleOp*%
_class
loc:@training/Adam/iter*
	container *
shape: *
dtype0	*
_output_shapes
: *#
shared_nametraining/Adam/iter
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
�
.training/Adam/beta_1/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *'
_class
loc:@training/Adam/beta_1*
valueB
 *fff?
�
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
�
training/Adam/beta_1/AssignAssignVariableOptraining/Adam/beta_1.training/Adam/beta_1/Initializer/initial_value*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
�
.training/Adam/beta_2/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *'
_class
loc:@training/Adam/beta_2*
valueB
 *w�?
�
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
�
training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
�
-training/Adam/decay/Initializer/initial_valueConst*&
_class
loc:@training/Adam/decay*
valueB
 *    *
dtype0*
_output_shapes
: 
�
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
�
5training/Adam/learning_rate/Initializer/initial_valueConst*.
_class$
" loc:@training/Adam/learning_rate*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
training/Adam/learning_rateVarHandleOp*,
shared_nametraining/Adam/learning_rate*.
_class$
" loc:@training/Adam/learning_rate*
	container *
shape: *
dtype0*
_output_shapes
: 
�
<training/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
�
"training/Adam/learning_rate/AssignAssignVariableOptraining/Adam/learning_rate5training/Adam/learning_rate/Initializer/initial_value*
dtype0
�
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
�
/training/Adam/conv2d/kernel/m/Initializer/zerosConst*%
valueB@*    * 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@
�
training/Adam/conv2d/kernel/mVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel
�
>training/Adam/conv2d/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/m*
_output_shapes
: * 
_class
loc:@conv2d/kernel
�
$training/Adam/conv2d/kernel/m/AssignAssignVariableOptraining/Adam/conv2d/kernel/m/training/Adam/conv2d/kernel/m/Initializer/zeros*
dtype0
�
1training/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/m* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@
�
-training/Adam/conv2d/bias/m/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
�
training/Adam/conv2d/bias/mVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias
�
<training/Adam/conv2d/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
_output_shapes
: 
�
"training/Adam/conv2d/bias/m/AssignAssignVariableOptraining/Adam/conv2d/bias/m-training/Adam/conv2d/bias/m/Initializer/zeros*
dtype0
�
/training/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/m*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
�
Atraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   *"
_class
loc:@conv2d_1/kernel
�
7training/Adam/conv2d_1/kernel/m/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *"
_class
loc:@conv2d_1/kernel
�
1training/Adam/conv2d_1/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/m/Initializer/zeros/Const*&
_output_shapes
:@@*
T0*

index_type0*"
_class
loc:@conv2d_1/kernel
�
training/Adam/conv2d_1/kernel/mVarHandleOp*0
shared_name!training/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
�
@training/Adam/conv2d_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
&training/Adam/conv2d_1/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/m1training/Adam/conv2d_1/kernel/m/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/m*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@
�
/training/Adam/conv2d_1/bias/m/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
�
training/Adam/conv2d_1/bias/mVarHandleOp* 
_class
loc:@conv2d_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/m
�
>training/Adam/conv2d_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/m* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
�
$training/Adam/conv2d_1/bias/m/AssignAssignVariableOptraining/Adam/conv2d_1/bias/m/training/Adam/conv2d_1/bias/m/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/m*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
Atraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
�
7training/Adam/conv2d_2/kernel/m/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
�
1training/Adam/conv2d_2/kernel/m/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/m/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/m/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
training/Adam/conv2d_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@@
�
@training/Adam/conv2d_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
�
&training/Adam/conv2d_2/kernel/m/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/m1training/Adam/conv2d_2/kernel/m/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/m*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@
�
/training/Adam/conv2d_2/bias/m/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
�
training/Adam/conv2d_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
	container *
shape:@
�
>training/Adam/conv2d_2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/m*
_output_shapes
: * 
_class
loc:@conv2d_2/bias
�
$training/Adam/conv2d_2/bias/m/AssignAssignVariableOptraining/Adam/conv2d_2/bias/m/training/Adam/conv2d_2/bias/m/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/m* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
�
>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensorConst*
valueB"@     *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
�
4training/Adam/dense/kernel/m/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
�
.training/Adam/dense/kernel/m/Initializer/zerosFill>training/Adam/dense/kernel/m/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/m/Initializer/zeros/Const*
_output_shapes
:	�*
T0*

index_type0*
_class
loc:@dense/kernel
�
training/Adam/dense/kernel/mVarHandleOp*-
shared_nametraining/Adam/dense/kernel/m*
_class
loc:@dense/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
: 
�
=training/Adam/dense/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/m*
_output_shapes
: *
_class
loc:@dense/kernel
�
#training/Adam/dense/kernel/m/AssignAssignVariableOptraining/Adam/dense/kernel/m.training/Adam/dense/kernel/m/Initializer/zeros*
dtype0
�
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
dtype0*
_output_shapes
:	�*
_class
loc:@dense/kernel
�
,training/Adam/dense/bias/m/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
�
training/Adam/dense/bias/mVarHandleOp*
dtype0*
_output_shapes
: *+
shared_nametraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
	container *
shape:
�
;training/Adam/dense/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
_output_shapes
: 
�
!training/Adam/dense/bias/m/AssignAssignVariableOptraining/Adam/dense/bias/m,training/Adam/dense/bias/m/Initializer/zeros*
dtype0
�
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
�
/training/Adam/conv2d/kernel/v/Initializer/zerosConst*%
valueB@*    * 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@
�
training/Adam/conv2d/kernel/vVarHandleOp* 
_class
loc:@conv2d/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d/kernel/v
�
>training/Adam/conv2d/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/kernel/v* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
$training/Adam/conv2d/kernel/v/AssignAssignVariableOptraining/Adam/conv2d/kernel/v/training/Adam/conv2d/kernel/v/Initializer/zeros*
dtype0
�
1training/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/v*
dtype0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel
�
-training/Adam/conv2d/bias/v/Initializer/zerosConst*
valueB@*    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
�
training/Adam/conv2d/bias/vVarHandleOp*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: *,
shared_nametraining/Adam/conv2d/bias/v
�
<training/Adam/conv2d/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
_output_shapes
: 
�
"training/Adam/conv2d/bias/v/AssignAssignVariableOptraining/Adam/conv2d/bias/v-training/Adam/conv2d/bias/v/Initializer/zeros*
dtype0
�
/training/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/v*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
�
Atraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   *"
_class
loc:@conv2d_1/kernel
�
7training/Adam/conv2d_1/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
�
1training/Adam/conv2d_1/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_1/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_1/kernel/v/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
�
training/Adam/conv2d_1/kernel/vVarHandleOp*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_1/kernel/v
�
@training/Adam/conv2d_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/kernel/v*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
&training/Adam/conv2d_1/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_1/kernel/v1training/Adam/conv2d_1/kernel/v/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/v*
dtype0*&
_output_shapes
:@@*"
_class
loc:@conv2d_1/kernel
�
/training/Adam/conv2d_1/bias/v/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    * 
_class
loc:@conv2d_1/bias
�
training/Adam/conv2d_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *.
shared_nametraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
	container *
shape:@
�
>training/Adam/conv2d_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_1/bias/v* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
�
$training/Adam/conv2d_1/bias/v/AssignAssignVariableOptraining/Adam/conv2d_1/bias/v/training/Adam/conv2d_1/bias/v/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/v*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
�
Atraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
�
7training/Adam/conv2d_2/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
�
1training/Adam/conv2d_2/kernel/v/Initializer/zerosFillAtraining/Adam/conv2d_2/kernel/v/Initializer/zeros/shape_as_tensor7training/Adam/conv2d_2/kernel/v/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@
�
training/Adam/conv2d_2/kernel/vVarHandleOp*
shape:@@*
dtype0*
_output_shapes
: *0
shared_name!training/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
	container 
�
@training/Adam/conv2d_2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
�
&training/Adam/conv2d_2/kernel/v/AssignAssignVariableOptraining/Adam/conv2d_2/kernel/v1training/Adam/conv2d_2/kernel/v/Initializer/zeros*
dtype0
�
3training/Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/kernel/v*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:@@
�
/training/Adam/conv2d_2/bias/v/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:@
�
training/Adam/conv2d_2/bias/vVarHandleOp*.
shared_nametraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
>training/Adam/conv2d_2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/conv2d_2/bias/v* 
_class
loc:@conv2d_2/bias*
_output_shapes
: 
�
$training/Adam/conv2d_2/bias/v/AssignAssignVariableOptraining/Adam/conv2d_2/bias/v/training/Adam/conv2d_2/bias/v/Initializer/zeros*
dtype0
�
1training/Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_2/bias/v*
dtype0*
_output_shapes
:@* 
_class
loc:@conv2d_2/bias
�
>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensorConst*
valueB"@     *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
�
4training/Adam/dense/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
�
.training/Adam/dense/kernel/v/Initializer/zerosFill>training/Adam/dense/kernel/v/Initializer/zeros/shape_as_tensor4training/Adam/dense/kernel/v/Initializer/zeros/Const*
_output_shapes
:	�*
T0*

index_type0*
_class
loc:@dense/kernel
�
training/Adam/dense/kernel/vVarHandleOp*-
shared_nametraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
: 
�
=training/Adam/dense/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/kernel/v*
_output_shapes
: *
_class
loc:@dense/kernel
�
#training/Adam/dense/kernel/v/AssignAssignVariableOptraining/Adam/dense/kernel/v.training/Adam/dense/kernel/v/Initializer/zeros*
dtype0
�
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	�
�
,training/Adam/dense/bias/v/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@dense/bias
�
training/Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *+
shared_nametraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
	container *
shape:
�
;training/Adam/dense/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense/bias/v*
_class
loc:@dense/bias*
_output_shapes
: 
�
!training/Adam/dense/bias/v/AssignAssignVariableOptraining/Adam/dense/bias/v,training/Adam/dense/bias/v/Initializer/zeros*
dtype0
�
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
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
T0	*
_output_shapes
: 
m
training/Adam/CastCasttraining/Adam/add*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0	
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
 *  �?*
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
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
 *���3*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/Identity_1*
_output_shapes
: *
T0
Z
training/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
T0*
_output_shapes
: 
�
9training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdamResourceApplyAdamconv2d/kerneltraining/Adam/conv2d/kernel/mtraining/Adam/conv2d/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstItraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0* 
_class
loc:@conv2d/kernel*
use_nesterov( 
�
7training/Adam/Adam/update_conv2d/bias/ResourceApplyAdamResourceApplyAdamconv2d/biastraining/Adam/conv2d/bias/mtraining/Adam/conv2d/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstAtraining/Adam/gradients/gradients/conv2d/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0*
_class
loc:@conv2d/bias
�
;training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdamResourceApplyAdamconv2d_1/kerneltraining/Adam/conv2d_1/kernel/mtraining/Adam/conv2d_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
use_nesterov( 
�
9training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdamResourceApplyAdamconv2d_1/biastraining/Adam/conv2d_1/bias/mtraining/Adam/conv2d_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0* 
_class
loc:@conv2d_1/bias
�
;training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdamResourceApplyAdamconv2d_2/kerneltraining/Adam/conv2d_2/kernel/mtraining/Adam/conv2d_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstKtraining/Adam/gradients/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
use_nesterov( 
�
9training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdamResourceApplyAdamconv2d_2/biastraining/Adam/conv2d_2/bias/mtraining/Adam/conv2d_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstCtraining/Adam/gradients/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0* 
_class
loc:@conv2d_2/bias
�
8training/Adam/Adam/update_dense/kernel/ResourceApplyAdamResourceApplyAdamdense/kerneltraining/Adam/dense/kernel/mtraining/Adam/dense/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1*
use_nesterov( *
use_locking(*
T0*
_class
loc:@dense/kernel
�
6training/Adam/Adam/update_dense/bias/ResourceApplyAdamResourceApplyAdam
dense/biastraining/Adam/dense/bias/mtraining/Adam/dense/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const@training/Adam/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense/bias*
use_nesterov( 
�
training/Adam/Adam/ConstConst8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: *
value	B	 R
x
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining/Adam/itertraining/Adam/Adam/Const*
dtype0	
�
!training/Adam/Adam/ReadVariableOpReadVariableOptraining/Adam/iter'^training/Adam/Adam/AssignAddVariableOp8^training/Adam/Adam/update_conv2d/bias/ResourceApplyAdam:^training/Adam/Adam/update_conv2d/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_1/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_1/kernel/ResourceApplyAdam:^training/Adam/Adam/update_conv2d_2/bias/ResourceApplyAdam<^training/Adam/Adam/update_conv2d_2/kernel/ResourceApplyAdam7^training/Adam/Adam/update_dense/bias/ResourceApplyAdam9^training/Adam/Adam/update_dense/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp
L
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 
E
AssignVariableOpAssignVariableOptotalPlaceholder*
dtype0
_
ReadVariableOpReadVariableOptotal^AssignVariableOp*
dtype0*
_output_shapes
: 
N
Placeholder_1Placeholder*
shape: *
dtype0*
_output_shapes
: 
I
AssignVariableOp_1AssignVariableOpcountPlaceholder_1*
dtype0
c
ReadVariableOp_1ReadVariableOpcount^AssignVariableOp_1*
dtype0*
_output_shapes
: 
T
VarIsInitializedOpVarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
S
VarIsInitializedOp_1VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_2VarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpdense/kernel*
_output_shapes
: 
N
VarIsInitializedOp_4VarIsInitializedOp
dense/bias*
_output_shapes
: 
I
VarIsInitializedOp_5VarIsInitializedOpcount*
_output_shapes
: 
W
VarIsInitializedOp_6VarIsInitializedOptraining/Adam/decay*
_output_shapes
: 
a
VarIsInitializedOp_7VarIsInitializedOptraining/Adam/conv2d/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_8VarIsInitializedOptraining/Adam/conv2d_1/bias/v*
_output_shapes
: 
`
VarIsInitializedOp_9VarIsInitializedOptraining/Adam/dense/kernel/v*
_output_shapes
: 
b
VarIsInitializedOp_10VarIsInitializedOptraining/Adam/conv2d/kernel/m*
_output_shapes
: 
Y
VarIsInitializedOp_11VarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
T
VarIsInitializedOp_12VarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
P
VarIsInitializedOp_13VarIsInitializedOpconv2d/bias*
_output_shapes
: 
J
VarIsInitializedOp_14VarIsInitializedOptotal*
_output_shapes
: 
d
VarIsInitializedOp_15VarIsInitializedOptraining/Adam/conv2d_1/kernel/m*
_output_shapes
: 
a
VarIsInitializedOp_16VarIsInitializedOptraining/Adam/dense/kernel/m*
_output_shapes
: 
_
VarIsInitializedOp_17VarIsInitializedOptraining/Adam/dense/bias/m*
_output_shapes
: 
d
VarIsInitializedOp_18VarIsInitializedOptraining/Adam/conv2d_2/kernel/v*
_output_shapes
: 
_
VarIsInitializedOp_19VarIsInitializedOptraining/Adam/dense/bias/v*
_output_shapes
: 
b
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/conv2d_1/bias/m*
_output_shapes
: 
R
VarIsInitializedOp_21VarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
d
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/conv2d_2/kernel/m*
_output_shapes
: 
`
VarIsInitializedOp_23VarIsInitializedOptraining/Adam/conv2d/bias/v*
_output_shapes
: 
d
VarIsInitializedOp_24VarIsInitializedOptraining/Adam/conv2d_1/kernel/v*
_output_shapes
: 
b
VarIsInitializedOp_25VarIsInitializedOptraining/Adam/conv2d_2/bias/v*
_output_shapes
: 
Y
VarIsInitializedOp_26VarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
`
VarIsInitializedOp_27VarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
`
VarIsInitializedOp_28VarIsInitializedOptraining/Adam/conv2d/bias/m*
_output_shapes
: 
R
VarIsInitializedOp_29VarIsInitializedOpconv2d/kernel*
_output_shapes
: 
b
VarIsInitializedOp_30VarIsInitializedOptraining/Adam/conv2d_2/bias/m*
_output_shapes
: 
�
initNoOp^conv2d/bias/Assign^conv2d/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^count/Assign^dense/bias/Assign^dense/kernel/Assign^total/Assign^training/Adam/beta_1/Assign^training/Adam/beta_2/Assign#^training/Adam/conv2d/bias/m/Assign#^training/Adam/conv2d/bias/v/Assign%^training/Adam/conv2d/kernel/m/Assign%^training/Adam/conv2d/kernel/v/Assign%^training/Adam/conv2d_1/bias/m/Assign%^training/Adam/conv2d_1/bias/v/Assign'^training/Adam/conv2d_1/kernel/m/Assign'^training/Adam/conv2d_1/kernel/v/Assign%^training/Adam/conv2d_2/bias/m/Assign%^training/Adam/conv2d_2/bias/v/Assign'^training/Adam/conv2d_2/kernel/m/Assign'^training/Adam/conv2d_2/kernel/v/Assign^training/Adam/decay/Assign"^training/Adam/dense/bias/m/Assign"^training/Adam/dense/bias/v/Assign$^training/Adam/dense/kernel/m/Assign$^training/Adam/dense/kernel/v/Assign^training/Adam/iter/Assign#^training/Adam/learning_rate/Assign
(
evaluation/group_depsNoOp	^loss/mul
�
conv2d_3_inputPlaceholder*$
shape:���������FF*
dtype0*/
_output_shapes
:���������FF
�
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_3/kernel*%
valueB"         @   
�
.conv2d_3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *�hϽ*
dtype0*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *�h�=*
dtype0*
_output_shapes
: 
�
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 
�
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_3/kernel
�
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:@
�
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:@
�
conv2d_3/kernelVarHandleOp* 
shared_nameconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
t
conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
conv2d_3/bias/Initializer/zerosConst* 
_class
loc:@conv2d_3/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_3/biasVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
	container 
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
e
conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros*
dtype0
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:@
g
conv2d_3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
conv2d_3/Conv2DConv2Dconv2d_3_inputconv2d_3/Conv2D/ReadVariableOp*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������DD@
i
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:@
�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:���������DD@*
T0
e
activation_4/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:���������DD@
�
max_pooling2d_3/MaxPoolMaxPoolactivation_4/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:���������""@
�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_4/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
.conv2d_4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *:͓�*
dtype0*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *:͓=*
dtype0*
_output_shapes
: 
�
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_4/kernel*
seed2 
�
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@
�
conv2d_4/kernelVarHandleOp* 
shared_nameconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
t
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_4/bias/Initializer/zerosConst* 
_class
loc:@conv2d_4/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_4/biasVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_4/bias* 
_class
loc:@conv2d_4/bias
k
.conv2d_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
e
conv2d_4/bias/AssignAssignVariableOpconv2d_4/biasconv2d_4/bias/Initializer/zeros*
dtype0
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:@
g
conv2d_4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_4/Conv2DConv2Dmax_pooling2d_3/MaxPoolconv2d_4/Conv2D/ReadVariableOp*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������  @
i
conv2d_4/BiasAdd/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:@
�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������  @
e
activation_5/ReluReluconv2d_4/BiasAdd*
T0*/
_output_shapes
:���������  @
�
max_pooling2d_4/MaxPoolMaxPoolactivation_5/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:���������@
�
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_5/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
�
.conv2d_5/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *:͓�*
dtype0*
_output_shapes
: 
�
.conv2d_5/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *:͓=*
dtype0*
_output_shapes
: 
�
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_5/kernel*
seed2 
�
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
�
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@
�
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*&
_output_shapes
:@@*
T0*"
_class
loc:@conv2d_5/kernel
�
conv2d_5/kernelVarHandleOp* 
shared_nameconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
o
0conv2d_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 
t
conv2d_5/kernel/AssignAssignVariableOpconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*
dtype0
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_5/bias/Initializer/zerosConst* 
_class
loc:@conv2d_5/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
conv2d_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
	container *
shape:@
k
.conv2d_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/bias*
_output_shapes
: 
e
conv2d_5/bias/AssignAssignVariableOpconv2d_5/biasconv2d_5/bias/Initializer/zeros*
dtype0
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
dtype0*
_output_shapes
:@
g
conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_5/Conv2D/ReadVariableOpReadVariableOpconv2d_5/kernel*
dtype0*&
_output_shapes
:@@
�
conv2d_5/Conv2DConv2Dmax_pooling2d_4/MaxPoolconv2d_5/Conv2D/ReadVariableOp*/
_output_shapes
:���������@*
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
i
conv2d_5/BiasAdd/ReadVariableOpReadVariableOpconv2d_5/bias*
dtype0*
_output_shapes
:@
�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2Dconv2d_5/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������@
e
activation_6/ReluReluconv2d_5/BiasAdd*
T0*/
_output_shapes
:���������@
�
max_pooling2d_5/MaxPoolMaxPoolactivation_6/Relu*
ksize
*
paddingVALID*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides

f
flatten_1/ShapeShapemax_pooling2d_5/MaxPool*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
d
flatten_1/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
flatten_1/Reshape/shapePackflatten_1/strided_sliceflatten_1/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
�
flatten_1/ReshapeReshapemax_pooling2d_5/MaxPoolflatten_1/Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
�
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_1/kernel*
valueB"@     
�
-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *�3�
�
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
valueB
 *�3=*
dtype0*
_output_shapes
: 
�
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	�*

seed 
�
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
�
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
:	�*
T0*!
_class
loc:@dense_1/kernel
�
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape:	�
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	�
�
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
dense_1/biasVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_namedense_1/bias*
_class
loc:@dense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
m
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/MatMul/ReadVariableOp*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:���������
b
activation_7/SoftmaxSoftmaxdense_1/BiasAdd*
T0*'
_output_shapes
:���������
�
activation_7_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
z
total_1/Initializer/zerosConst*
_class
loc:@total_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
total_1VarHandleOp*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name	total_1*
_class
loc:@total_1
_
(total_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
S
total_1/AssignAssignVariableOptotal_1total_1/Initializer/zeros*
dtype0
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: 
z
count_1/Initializer/zerosConst*
_class
loc:@count_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
count_1VarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_name	count_1*
_class
loc:@count_1*
	container 
_
(count_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_1*
_output_shapes
: 
S
count_1/AssignAssignVariableOpcount_1count_1/Initializer/zeros*
dtype0
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
dtype0*
_output_shapes
: 
�
metrics_2/acc/SqueezeSqueezeactivation_7_target*
T0*#
_output_shapes
:���������*
squeeze_dims

���������
i
metrics_2/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics_2/acc/ArgMaxArgMaxactivation_7/Softmaxmetrics_2/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
}
metrics_2/acc/CastCastmetrics_2/acc/ArgMax*
Truncate( *

DstT0*#
_output_shapes
:���������*

SrcT0	
�
metrics_2/acc/EqualEqualmetrics_2/acc/Squeezemetrics_2/acc/Cast*
T0*#
_output_shapes
:���������*
incompatible_shape_error(
~
metrics_2/acc/Cast_1Castmetrics_2/acc/Equal*
Truncate( *

DstT0*#
_output_shapes
:���������*

SrcT0

]
metrics_2/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
metrics_2/acc/SumSummetrics_2/acc/Cast_1metrics_2/acc/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
a
!metrics_2/acc/AssignAddVariableOpAssignAddVariableOptotal_1metrics_2/acc/Sum*
dtype0
�
metrics_2/acc/ReadVariableOpReadVariableOptotal_1"^metrics_2/acc/AssignAddVariableOp^metrics_2/acc/Sum*
dtype0*
_output_shapes
: 
a
metrics_2/acc/SizeSizemetrics_2/acc/Cast_1*
_output_shapes
: *
T0*
out_type0
p
metrics_2/acc/Cast_2Castmetrics_2/acc/Size*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
�
#metrics_2/acc/AssignAddVariableOp_1AssignAddVariableOpcount_1metrics_2/acc/Cast_2"^metrics_2/acc/AssignAddVariableOp*
dtype0
�
metrics_2/acc/ReadVariableOp_1ReadVariableOpcount_1"^metrics_2/acc/AssignAddVariableOp$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics_2/acc/div_no_nan/ReadVariableOpReadVariableOptotal_1$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
)metrics_2/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount_1$^metrics_2/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics_2/acc/div_no_nanDivNoNan'metrics_2/acc/div_no_nan/ReadVariableOp)metrics_2/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
]
metrics_2/acc/IdentityIdentitymetrics_2/acc/div_no_nan*
T0*
_output_shapes
: 
�
loss_1/activation_7_loss/CastCastactivation_7_target*
Truncate( *

DstT0	*0
_output_shapes
:������������������*

SrcT0
m
loss_1/activation_7_loss/ShapeShapedense_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
y
&loss_1/activation_7_loss/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
 loss_1/activation_7_loss/ReshapeReshapeloss_1/activation_7_loss/Cast&loss_1/activation_7_loss/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:���������

,loss_1/activation_7_loss/strided_slice/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:
x
.loss_1/activation_7_loss/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
x
.loss_1/activation_7_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
&loss_1/activation_7_loss/strided_sliceStridedSliceloss_1/activation_7_loss/Shape,loss_1/activation_7_loss/strided_slice/stack.loss_1/activation_7_loss/strided_slice/stack_1.loss_1/activation_7_loss/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
u
*loss_1/activation_7_loss/Reshape_1/shape/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
�
(loss_1/activation_7_loss/Reshape_1/shapePack*loss_1/activation_7_loss/Reshape_1/shape/0&loss_1/activation_7_loss/strided_slice*
T0*

axis *
N*
_output_shapes
:
�
"loss_1/activation_7_loss/Reshape_1Reshapedense_1/BiasAdd(loss_1/activation_7_loss/Reshape_1/shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
Bloss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShape loss_1/activation_7_loss/Reshape*
T0	*
out_type0*
_output_shapes
:
�
`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits"loss_1/activation_7_loss/Reshape_1 loss_1/activation_7_loss/Reshape*
Tlabels0	*?
_output_shapes-
+:���������:������������������*
T0
r
-loss_1/activation_7_loss/weighted_loss/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
[loss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Zloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
Zloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
Yloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
q
iloss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Hloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsj^loss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Hloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/ConstConstj^loss_1/activation_7_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_likeFillHloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/ShapeHloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:���������
�
8loss_1/activation_7_loss/weighted_loss/broadcast_weightsMul-loss_1/activation_7_loss/weighted_loss/Cast/xBloss_1/activation_7_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
*loss_1/activation_7_loss/weighted_loss/MulMul`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits8loss_1/activation_7_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
h
loss_1/activation_7_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
loss_1/activation_7_loss/SumSum*loss_1/activation_7_loss/weighted_loss/Mulloss_1/activation_7_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
%loss_1/activation_7_loss/num_elementsSize*loss_1/activation_7_loss/weighted_loss/Mul*
_output_shapes
: *
T0*
out_type0
�
*loss_1/activation_7_loss/num_elements/CastCast%loss_1/activation_7_loss/num_elements*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
c
 loss_1/activation_7_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
�
loss_1/activation_7_loss/Sum_1Sumloss_1/activation_7_loss/Sum loss_1/activation_7_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
loss_1/activation_7_loss/valueDivNoNanloss_1/activation_7_loss/Sum_1*loss_1/activation_7_loss/num_elements/Cast*
_output_shapes
: *
T0
Q
loss_1/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
`

loss_1/mulMulloss_1/mul/xloss_1/activation_7_loss/value*
T0*
_output_shapes
: 
l
)training_2/Adam/gradients/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
r
-training_2/Adam/gradients/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
(training_2/Adam/gradients/gradients/FillFill)training_2/Adam/gradients/gradients/Shape-training_2/Adam/gradients/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
7training_2/Adam/gradients/gradients/loss_1/mul_grad/MulMul(training_2/Adam/gradients/gradients/Fillloss_1/activation_7_loss/value*
T0*
_output_shapes
: 
�
9training_2/Adam/gradients/gradients/loss_1/mul_grad/Mul_1Mul(training_2/Adam/gradients/gradients/Fillloss_1/mul/x*
T0*
_output_shapes
: 
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
]training_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ShapeOtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Rtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nanDivNoNan9training_2/Adam/gradients/gradients/loss_1/mul_grad/Mul_1*loss_1/activation_7_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/SumSumRtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan]training_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/BroadcastGradientArgs*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ReshapeReshapeKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/SumMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/NegNegloss_1/activation_7_loss/Sum_1*
T0*
_output_shapes
: 
�
Ttraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_1DivNoNanKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Neg*loss_1/activation_7_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ttraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_2DivNoNanTtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_1*loss_1/activation_7_loss/num_elements/Cast*
T0*
_output_shapes
: 
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/mulMul9training_2/Adam/gradients/gradients/loss_1/mul_grad/Mul_1Ttraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/div_no_nan_2*
_output_shapes
: *
T0
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Sum_1SumKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/mul_training_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Qtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Reshape_1ReshapeMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Sum_1Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Utraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Otraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/ReshapeReshapeOtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/value_grad/ReshapeUtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
�
Ltraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/TileTileOtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/ReshapeMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
�
Straining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
Mtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/ReshapeReshapeLtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_1_grad/TileStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
�
Ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/ShapeShape*loss_1/activation_7_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
�
Jtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/TileTileMtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/ReshapeKtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
�
Ytraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/ShapeShape`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape_1Shape8loss_1/activation_7_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
�
itraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Wtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/MulMulJtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Tile8loss_1/activation_7_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
�
Wtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/SumSumWtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Mulitraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/ReshapeReshapeWtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/SumYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
Ytraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Mul_1Mul`loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsJtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Sum_grad/Tile*
T0*#
_output_shapes
:���������
�
Ytraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Sum_1SumYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Mul_1ktraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
]training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Reshape_1ReshapeYtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Sum_1[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
.training_2/Adam/gradients/gradients/zeros_like	ZerosLikebloss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientbloss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims[training_2/Adam/gradients/gradients/loss_1/activation_7_loss/weighted_loss/Mul_grad/Reshape�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*0
_output_shapes
:������������������
�
Qtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/ShapeShapedense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Straining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/ReshapeReshape�training_2/Adam/gradients/gradients/loss_1/activation_7_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulQtraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Dtraining_2/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
�
>training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMulMatMulStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Reshapedense_1/MatMul/ReadVariableOp*
T0*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
@training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/ReshapeStraining_2/Adam/gradients/gradients/loss_1/activation_7_loss/Reshape_1_grad/Reshape*
T0*
transpose_a(*
_output_shapes
:	�*
transpose_b( 
�
@training_2/Adam/gradients/gradients/flatten_1/Reshape_grad/ShapeShapemax_pooling2d_5/MaxPool*
T0*
out_type0*
_output_shapes
:
�
Btraining_2/Adam/gradients/gradients/flatten_1/Reshape_grad/ReshapeReshape>training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul@training_2/Adam/gradients/gradients/flatten_1/Reshape_grad/Shape*/
_output_shapes
:���������@*
T0*
Tshape0
�
Ltraining_2/Adam/gradients/gradients/max_pooling2d_5/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_6/Relumax_pooling2d_5/MaxPoolBtraining_2/Adam/gradients/gradients/flatten_1/Reshape_grad/Reshape*/
_output_shapes
:���������@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
Ctraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGradReluGradLtraining_2/Adam/gradients/gradients/max_pooling2d_5/MaxPool_grad/MaxPoolGradactivation_6/Relu*/
_output_shapes
:���������@*
T0
�
Etraining_2/Adam/gradients/gradients/conv2d_5/BiasAdd_grad/BiasAddGradBiasAddGradCtraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
?training_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/ShapeNShapeNmax_pooling2d_4/MaxPoolconv2d_5/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Ltraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_5/Conv2D/ReadVariableOpCtraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGrad*
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
:���������@
�
Mtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_4/MaxPoolAtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/ShapeN:1Ctraining_2/Adam/gradients/gradients/activation_6/Relu_grad/ReluGrad*
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
�
Ltraining_2/Adam/gradients/gradients/max_pooling2d_4/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_5/Relumax_pooling2d_4/MaxPoolLtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:���������  @*
T0*
data_formatNHWC*
strides

�
Ctraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGradReluGradLtraining_2/Adam/gradients/gradients/max_pooling2d_4/MaxPool_grad/MaxPoolGradactivation_5/Relu*
T0*/
_output_shapes
:���������  @
�
Etraining_2/Adam/gradients/gradients/conv2d_4/BiasAdd_grad/BiasAddGradBiasAddGradCtraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
?training_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/ShapeNShapeNmax_pooling2d_3/MaxPoolconv2d_4/Conv2D/ReadVariableOp*
N* 
_output_shapes
::*
T0*
out_type0
�
Ltraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_4/Conv2D/ReadVariableOpCtraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������""@*
	dilations
*
T0
�
Mtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_3/MaxPoolAtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/ShapeN:1Ctraining_2/Adam/gradients/gradients/activation_5/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@@*
	dilations
*
T0
�
Ltraining_2/Adam/gradients/gradients/max_pooling2d_3/MaxPool_grad/MaxPoolGradMaxPoolGradactivation_4/Relumax_pooling2d_3/MaxPoolLtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*/
_output_shapes
:���������DD@*
T0*
data_formatNHWC*
strides

�
Ctraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGradReluGradLtraining_2/Adam/gradients/gradients/max_pooling2d_3/MaxPool_grad/MaxPoolGradactivation_4/Relu*
T0*/
_output_shapes
:���������DD@
�
Etraining_2/Adam/gradients/gradients/conv2d_3/BiasAdd_grad/BiasAddGradBiasAddGradCtraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
?training_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/ShapeNShapeNconv2d_3_inputconv2d_3/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Ltraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?training_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_3/Conv2D/ReadVariableOpCtraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGrad*
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
:���������FF
�
Mtraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2d_3_inputAtraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/ShapeN:1Ctraining_2/Adam/gradients/gradients/activation_4/Relu_grad/ReluGrad*
paddingVALID*&
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
�
&training_2/Adam/iter/Initializer/zerosConst*'
_class
loc:@training_2/Adam/iter*
value	B	 R *
dtype0	*
_output_shapes
: 
�
training_2/Adam/iterVarHandleOp*%
shared_nametraining_2/Adam/iter*'
_class
loc:@training_2/Adam/iter*
	container *
shape: *
dtype0	*
_output_shapes
: 
y
5training_2/Adam/iter/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/iter*
_output_shapes
: 
z
training_2/Adam/iter/AssignAssignVariableOptraining_2/Adam/iter&training_2/Adam/iter/Initializer/zeros*
dtype0	
u
(training_2/Adam/iter/Read/ReadVariableOpReadVariableOptraining_2/Adam/iter*
dtype0	*
_output_shapes
: 
�
0training_2/Adam/beta_1/Initializer/initial_valueConst*)
_class
loc:@training_2/Adam/beta_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
training_2/Adam/beta_1VarHandleOp*'
shared_nametraining_2/Adam/beta_1*)
_class
loc:@training_2/Adam/beta_1*
	container *
shape: *
dtype0*
_output_shapes
: 
}
7training_2/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/beta_1*
_output_shapes
: 
�
training_2/Adam/beta_1/AssignAssignVariableOptraining_2/Adam/beta_10training_2/Adam/beta_1/Initializer/initial_value*
dtype0
y
*training_2/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
dtype0*
_output_shapes
: 
�
0training_2/Adam/beta_2/Initializer/initial_valueConst*)
_class
loc:@training_2/Adam/beta_2*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
training_2/Adam/beta_2VarHandleOp*)
_class
loc:@training_2/Adam/beta_2*
	container *
shape: *
dtype0*
_output_shapes
: *'
shared_nametraining_2/Adam/beta_2
}
7training_2/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/beta_2*
_output_shapes
: 
�
training_2/Adam/beta_2/AssignAssignVariableOptraining_2/Adam/beta_20training_2/Adam/beta_2/Initializer/initial_value*
dtype0
y
*training_2/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
dtype0*
_output_shapes
: 
�
/training_2/Adam/decay/Initializer/initial_valueConst*(
_class
loc:@training_2/Adam/decay*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training_2/Adam/decayVarHandleOp*
shape: *
dtype0*
_output_shapes
: *&
shared_nametraining_2/Adam/decay*(
_class
loc:@training_2/Adam/decay*
	container 
{
6training_2/Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/decay*
_output_shapes
: 
�
training_2/Adam/decay/AssignAssignVariableOptraining_2/Adam/decay/training_2/Adam/decay/Initializer/initial_value*
dtype0
w
)training_2/Adam/decay/Read/ReadVariableOpReadVariableOptraining_2/Adam/decay*
dtype0*
_output_shapes
: 
�
7training_2/Adam/learning_rate/Initializer/initial_valueConst*0
_class&
$"loc:@training_2/Adam/learning_rate*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
training_2/Adam/learning_rateVarHandleOp*
shape: *
dtype0*
_output_shapes
: *.
shared_nametraining_2/Adam/learning_rate*0
_class&
$"loc:@training_2/Adam/learning_rate*
	container 
�
>training_2/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/learning_rate*
_output_shapes
: 
�
$training_2/Adam/learning_rate/AssignAssignVariableOptraining_2/Adam/learning_rate7training_2/Adam/learning_rate/Initializer/initial_value*
dtype0
�
1training_2/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_3/kernel/m/Initializer/zerosConst*%
valueB@*    *"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
!training_2/Adam/conv2d_3/kernel/mVarHandleOp*2
shared_name#!training_2/Adam/conv2d_3/kernel/m*"
_class
loc:@conv2d_3/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
Btraining_2/Adam/conv2d_3/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_3/kernel/m*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_3/kernel/m/AssignAssignVariableOp!training_2/Adam/conv2d_3/kernel/m3training_2/Adam/conv2d_3/kernel/m/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_3/kernel/m*"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
1training_2/Adam/conv2d_3/bias/m/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_3/bias/mVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_3/bias/m* 
_class
loc:@conv2d_3/bias*
	container *
shape:@
�
@training_2/Adam/conv2d_3/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_3/bias/m* 
_class
loc:@conv2d_3/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_3/bias/m/AssignAssignVariableOptraining_2/Adam/conv2d_3/bias/m1training_2/Adam/conv2d_3/bias/m/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_3/bias/m* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:@
�
Ctraining_2/Adam/conv2d_4/kernel/m/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
:
�
9training_2/Adam/conv2d_4/kernel/m/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_4/kernel/m/Initializer/zerosFillCtraining_2/Adam/conv2d_4/kernel/m/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_4/kernel/m/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@
�
!training_2/Adam/conv2d_4/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_4/kernel/m*"
_class
loc:@conv2d_4/kernel*
	container *
shape:@@
�
Btraining_2/Adam/conv2d_4/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_4/kernel/m*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_4/kernel/m/AssignAssignVariableOp!training_2/Adam/conv2d_4/kernel/m3training_2/Adam/conv2d_4/kernel/m/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_4/kernel/m*"
_class
loc:@conv2d_4/kernel*
dtype0*&
_output_shapes
:@@
�
1training_2/Adam/conv2d_4/bias/m/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_4/bias/mVarHandleOp*0
shared_name!training_2/Adam/conv2d_4/bias/m* 
_class
loc:@conv2d_4/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
�
@training_2/Adam/conv2d_4/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_4/bias/m* 
_class
loc:@conv2d_4/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_4/bias/m/AssignAssignVariableOptraining_2/Adam/conv2d_4/bias/m1training_2/Adam/conv2d_4/bias/m/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_4/bias/m* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:@
�
Ctraining_2/Adam/conv2d_5/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   *"
_class
loc:@conv2d_5/kernel
�
9training_2/Adam/conv2d_5/kernel/m/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_5/kernel/m/Initializer/zerosFillCtraining_2/Adam/conv2d_5/kernel/m/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_5/kernel/m/Initializer/zeros/Const*&
_output_shapes
:@@*
T0*

index_type0*"
_class
loc:@conv2d_5/kernel
�
!training_2/Adam/conv2d_5/kernel/mVarHandleOp*"
_class
loc:@conv2d_5/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_5/kernel/m
�
Btraining_2/Adam/conv2d_5/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_5/kernel/m*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_5/kernel/m/AssignAssignVariableOp!training_2/Adam/conv2d_5/kernel/m3training_2/Adam/conv2d_5/kernel/m/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_5/kernel/m*"
_class
loc:@conv2d_5/kernel*
dtype0*&
_output_shapes
:@@
�
1training_2/Adam/conv2d_5/bias/m/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_5/bias/mVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_5/bias/m* 
_class
loc:@conv2d_5/bias*
	container *
shape:@
�
@training_2/Adam/conv2d_5/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_5/bias/m* 
_class
loc:@conv2d_5/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_5/bias/m/AssignAssignVariableOptraining_2/Adam/conv2d_5/bias/m1training_2/Adam/conv2d_5/bias/m/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_5/bias/m* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
:@
�
Btraining_2/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensorConst*
valueB"@     *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
�
8training_2/Adam/dense_1/kernel/m/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
�
2training_2/Adam/dense_1/kernel/m/Initializer/zerosFillBtraining_2/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensor8training_2/Adam/dense_1/kernel/m/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
�
 training_2/Adam/dense_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *1
shared_name" training_2/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
	container *
shape:	�
�
Atraining_2/Adam/dense_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOp training_2/Adam/dense_1/kernel/m*
_output_shapes
: *!
_class
loc:@dense_1/kernel
�
'training_2/Adam/dense_1/kernel/m/AssignAssignVariableOp training_2/Adam/dense_1/kernel/m2training_2/Adam/dense_1/kernel/m/Initializer/zeros*
dtype0
�
4training_2/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:	�
�
0training_2/Adam/dense_1/bias/m/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
�
training_2/Adam/dense_1/bias/mVarHandleOp*
_class
loc:@dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: */
shared_name training_2/Adam/dense_1/bias/m
�
?training_2/Adam/dense_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
_output_shapes
: 
�
%training_2/Adam/dense_1/bias/m/AssignAssignVariableOptraining_2/Adam/dense_1/bias/m0training_2/Adam/dense_1/bias/m/Initializer/zeros*
dtype0
�
2training_2/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
�
3training_2/Adam/conv2d_3/kernel/v/Initializer/zerosConst*%
valueB@*    *"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
!training_2/Adam/conv2d_3/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_3/kernel/v*"
_class
loc:@conv2d_3/kernel*
	container *
shape:@
�
Btraining_2/Adam/conv2d_3/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_3/kernel/v*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_3/kernel/v/AssignAssignVariableOp!training_2/Adam/conv2d_3/kernel/v3training_2/Adam/conv2d_3/kernel/v/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_3/kernel/v*"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:@
�
1training_2/Adam/conv2d_3/bias/v/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    * 
_class
loc:@conv2d_3/bias
�
training_2/Adam/conv2d_3/bias/vVarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_3/bias/v* 
_class
loc:@conv2d_3/bias*
	container *
shape:@
�
@training_2/Adam/conv2d_3/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_3/bias/v* 
_class
loc:@conv2d_3/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_3/bias/v/AssignAssignVariableOptraining_2/Adam/conv2d_3/bias/v1training_2/Adam/conv2d_3/bias/v/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_3/bias/v* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:@
�
Ctraining_2/Adam/conv2d_4/kernel/v/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @   @   *"
_class
loc:@conv2d_4/kernel
�
9training_2/Adam/conv2d_4/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_4/kernel/v/Initializer/zerosFillCtraining_2/Adam/conv2d_4/kernel/v/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_4/kernel/v/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@
�
!training_2/Adam/conv2d_4/kernel/vVarHandleOp*"
_class
loc:@conv2d_4/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_4/kernel/v
�
Btraining_2/Adam/conv2d_4/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_4/kernel/v*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel
�
(training_2/Adam/conv2d_4/kernel/v/AssignAssignVariableOp!training_2/Adam/conv2d_4/kernel/v3training_2/Adam/conv2d_4/kernel/v/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_4/kernel/v*"
_class
loc:@conv2d_4/kernel*
dtype0*&
_output_shapes
:@@
�
1training_2/Adam/conv2d_4/bias/v/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_4/bias/vVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_4/bias/v* 
_class
loc:@conv2d_4/bias
�
@training_2/Adam/conv2d_4/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_4/bias/v* 
_class
loc:@conv2d_4/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_4/bias/v/AssignAssignVariableOptraining_2/Adam/conv2d_4/bias/v1training_2/Adam/conv2d_4/bias/v/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_4/bias/v* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:@
�
Ctraining_2/Adam/conv2d_5/kernel/v/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
:
�
9training_2/Adam/conv2d_5/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 
�
3training_2/Adam/conv2d_5/kernel/v/Initializer/zerosFillCtraining_2/Adam/conv2d_5/kernel/v/Initializer/zeros/shape_as_tensor9training_2/Adam/conv2d_5/kernel/v/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@
�
!training_2/Adam/conv2d_5/kernel/vVarHandleOp*"
_class
loc:@conv2d_5/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: *2
shared_name#!training_2/Adam/conv2d_5/kernel/v
�
Btraining_2/Adam/conv2d_5/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp!training_2/Adam/conv2d_5/kernel/v*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
�
(training_2/Adam/conv2d_5/kernel/v/AssignAssignVariableOp!training_2/Adam/conv2d_5/kernel/v3training_2/Adam/conv2d_5/kernel/v/Initializer/zeros*
dtype0
�
5training_2/Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_5/kernel/v*
dtype0*&
_output_shapes
:@@*"
_class
loc:@conv2d_5/kernel
�
1training_2/Adam/conv2d_5/bias/v/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
:@
�
training_2/Adam/conv2d_5/bias/vVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *0
shared_name!training_2/Adam/conv2d_5/bias/v* 
_class
loc:@conv2d_5/bias
�
@training_2/Adam/conv2d_5/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/conv2d_5/bias/v* 
_class
loc:@conv2d_5/bias*
_output_shapes
: 
�
&training_2/Adam/conv2d_5/bias/v/AssignAssignVariableOptraining_2/Adam/conv2d_5/bias/v1training_2/Adam/conv2d_5/bias/v/Initializer/zeros*
dtype0
�
3training_2/Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_5/bias/v* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
:@
�
Btraining_2/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensorConst*
valueB"@     *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
�
8training_2/Adam/dense_1/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
�
2training_2/Adam/dense_1/kernel/v/Initializer/zerosFillBtraining_2/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensor8training_2/Adam/dense_1/kernel/v/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
�
 training_2/Adam/dense_1/kernel/vVarHandleOp*
shape:	�*
dtype0*
_output_shapes
: *1
shared_name" training_2/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
	container 
�
Atraining_2/Adam/dense_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOp training_2/Adam/dense_1/kernel/v*
_output_shapes
: *!
_class
loc:@dense_1/kernel
�
'training_2/Adam/dense_1/kernel/v/AssignAssignVariableOp training_2/Adam/dense_1/kernel/v2training_2/Adam/dense_1/kernel/v/Initializer/zeros*
dtype0
�
4training_2/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:	�
�
0training_2/Adam/dense_1/bias/v/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
�
training_2/Adam/dense_1/bias/vVarHandleOp*
shape:*
dtype0*
_output_shapes
: */
shared_name training_2/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
	container 
�
?training_2/Adam/dense_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
_output_shapes
: 
�
%training_2/Adam/dense_1/bias/v/AssignAssignVariableOptraining_2/Adam/dense_1/bias/v0training_2/Adam/dense_1/bias/v/Initializer/zeros*
dtype0
�
2training_2/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
}
'training_2/Adam/Identity/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
dtype0*
_output_shapes
: 
n
training_2/Adam/IdentityIdentity'training_2/Adam/Identity/ReadVariableOp*
T0*
_output_shapes
: 
k
training_2/Adam/ReadVariableOpReadVariableOptraining_2/Adam/iter*
dtype0	*
_output_shapes
: 
W
training_2/Adam/add/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
t
training_2/Adam/addAddV2training_2/Adam/ReadVariableOptraining_2/Adam/add/y*
T0	*
_output_shapes
: 
q
training_2/Adam/CastCasttraining_2/Adam/add*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
x
)training_2/Adam/Identity_1/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
dtype0*
_output_shapes
: 
r
training_2/Adam/Identity_1Identity)training_2/Adam/Identity_1/ReadVariableOp*
T0*
_output_shapes
: 
x
)training_2/Adam/Identity_2/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
dtype0*
_output_shapes
: 
r
training_2/Adam/Identity_2Identity)training_2/Adam/Identity_2/ReadVariableOp*
T0*
_output_shapes
: 
m
training_2/Adam/PowPowtraining_2/Adam/Identity_1training_2/Adam/Cast*
T0*
_output_shapes
: 
o
training_2/Adam/Pow_1Powtraining_2/Adam/Identity_2training_2/Adam/Cast*
_output_shapes
: *
T0
Z
training_2/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
i
training_2/Adam/subSubtraining_2/Adam/sub/xtraining_2/Adam/Pow_1*
T0*
_output_shapes
: 
R
training_2/Adam/SqrtSqrttraining_2/Adam/sub*
T0*
_output_shapes
: 
\
training_2/Adam/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
training_2/Adam/sub_1Subtraining_2/Adam/sub_1/xtraining_2/Adam/Pow*
T0*
_output_shapes
: 
p
training_2/Adam/truedivRealDivtraining_2/Adam/Sqrttraining_2/Adam/sub_1*
_output_shapes
: *
T0
n
training_2/Adam/mulMultraining_2/Adam/Identitytraining_2/Adam/truediv*
T0*
_output_shapes
: 
Z
training_2/Adam/ConstConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
\
training_2/Adam/sub_2/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
r
training_2/Adam/sub_2Subtraining_2/Adam/sub_2/xtraining_2/Adam/Identity_1*
_output_shapes
: *
T0
\
training_2/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
r
training_2/Adam/sub_3Subtraining_2/Adam/sub_3/xtraining_2/Adam/Identity_2*
T0*
_output_shapes
: 
�
=training_2/Adam/Adam/update_conv2d_3/kernel/ResourceApplyAdamResourceApplyAdamconv2d_3/kernel!training_2/Adam/conv2d_3/kernel/m!training_2/Adam/conv2d_3/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstMtraining_2/Adam/gradients/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*
T0*"
_class
loc:@conv2d_3/kernel*
use_nesterov( *
use_locking(
�
;training_2/Adam/Adam/update_conv2d_3/bias/ResourceApplyAdamResourceApplyAdamconv2d_3/biastraining_2/Adam/conv2d_3/bias/mtraining_2/Adam/conv2d_3/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstEtraining_2/Adam/gradients/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_3/bias*
use_nesterov( 
�
=training_2/Adam/Adam/update_conv2d_4/kernel/ResourceApplyAdamResourceApplyAdamconv2d_4/kernel!training_2/Adam/conv2d_4/kernel/m!training_2/Adam/conv2d_4/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstMtraining_2/Adam/gradients/gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*
use_nesterov( *
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel
�
;training_2/Adam/Adam/update_conv2d_4/bias/ResourceApplyAdamResourceApplyAdamconv2d_4/biastraining_2/Adam/conv2d_4/bias/mtraining_2/Adam/conv2d_4/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstEtraining_2/Adam/gradients/gradients/conv2d_4/BiasAdd_grad/BiasAddGrad*
use_nesterov( *
use_locking(*
T0* 
_class
loc:@conv2d_4/bias
�
=training_2/Adam/Adam/update_conv2d_5/kernel/ResourceApplyAdamResourceApplyAdamconv2d_5/kernel!training_2/Adam/conv2d_5/kernel/m!training_2/Adam/conv2d_5/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstMtraining_2/Adam/gradients/gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
use_nesterov( 
�
;training_2/Adam/Adam/update_conv2d_5/bias/ResourceApplyAdamResourceApplyAdamconv2d_5/biastraining_2/Adam/conv2d_5/bias/mtraining_2/Adam/conv2d_5/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstEtraining_2/Adam/gradients/gradients/conv2d_5/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
use_nesterov( 
�
<training_2/Adam/Adam/update_dense_1/kernel/ResourceApplyAdamResourceApplyAdamdense_1/kernel training_2/Adam/dense_1/kernel/m training_2/Adam/dense_1/kernel/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/Const@training_2/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
use_nesterov( 
�
:training_2/Adam/Adam/update_dense_1/bias/ResourceApplyAdamResourceApplyAdamdense_1/biastraining_2/Adam/dense_1/bias/mtraining_2/Adam/dense_1/bias/vtraining_2/Adam/Powtraining_2/Adam/Pow_1training_2/Adam/Identitytraining_2/Adam/Identity_1training_2/Adam/Identity_2training_2/Adam/ConstDtraining_2/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense_1/bias*
use_nesterov( 
�
training_2/Adam/Adam/ConstConst<^training_2/Adam/Adam/update_conv2d_3/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_3/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_4/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_4/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_5/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_5/kernel/ResourceApplyAdam;^training_2/Adam/Adam/update_dense_1/bias/ResourceApplyAdam=^training_2/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: *
value	B	 R
~
(training_2/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining_2/Adam/itertraining_2/Adam/Adam/Const*
dtype0	
�
#training_2/Adam/Adam/ReadVariableOpReadVariableOptraining_2/Adam/iter)^training_2/Adam/Adam/AssignAddVariableOp<^training_2/Adam/Adam/update_conv2d_3/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_3/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_4/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_4/kernel/ResourceApplyAdam<^training_2/Adam/Adam/update_conv2d_5/bias/ResourceApplyAdam>^training_2/Adam/Adam/update_conv2d_5/kernel/ResourceApplyAdam;^training_2/Adam/Adam/update_dense_1/bias/ResourceApplyAdam=^training_2/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
U
training_3/group_depsNoOp^loss_1/mul)^training_2/Adam/Adam/AssignAddVariableOp"�"�
trainable_variables��
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
�
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
�
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
�
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
�
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
�
conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
s
conv2d_5/bias:0conv2d_5/bias/Assign#conv2d_5/bias/Read/ReadVariableOp:0(2!conv2d_5/bias/Initializer/zeros:08
�
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"�J
	variables�J�J
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
�
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
�
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
�
training/Adam/iter:0training/Adam/iter/Assign(training/Adam/iter/Read/ReadVariableOp:0(2&training/Adam/iter/Initializer/zeros:0H
�
training/Adam/beta_1:0training/Adam/beta_1/Assign*training/Adam/beta_1/Read/ReadVariableOp:0(20training/Adam/beta_1/Initializer/initial_value:0H
�
training/Adam/beta_2:0training/Adam/beta_2/Assign*training/Adam/beta_2/Read/ReadVariableOp:0(20training/Adam/beta_2/Initializer/initial_value:0H
�
training/Adam/decay:0training/Adam/decay/Assign)training/Adam/decay/Read/ReadVariableOp:0(2/training/Adam/decay/Initializer/initial_value:0H
�
training/Adam/learning_rate:0"training/Adam/learning_rate/Assign1training/Adam/learning_rate/Read/ReadVariableOp:0(27training/Adam/learning_rate/Initializer/initial_value:0H
�
training/Adam/conv2d/kernel/m:0$training/Adam/conv2d/kernel/m/Assign3training/Adam/conv2d/kernel/m/Read/ReadVariableOp:0(21training/Adam/conv2d/kernel/m/Initializer/zeros:0
�
training/Adam/conv2d/bias/m:0"training/Adam/conv2d/bias/m/Assign1training/Adam/conv2d/bias/m/Read/ReadVariableOp:0(2/training/Adam/conv2d/bias/m/Initializer/zeros:0
�
!training/Adam/conv2d_1/kernel/m:0&training/Adam/conv2d_1/kernel/m/Assign5training/Adam/conv2d_1/kernel/m/Read/ReadVariableOp:0(23training/Adam/conv2d_1/kernel/m/Initializer/zeros:0
�
training/Adam/conv2d_1/bias/m:0$training/Adam/conv2d_1/bias/m/Assign3training/Adam/conv2d_1/bias/m/Read/ReadVariableOp:0(21training/Adam/conv2d_1/bias/m/Initializer/zeros:0
�
!training/Adam/conv2d_2/kernel/m:0&training/Adam/conv2d_2/kernel/m/Assign5training/Adam/conv2d_2/kernel/m/Read/ReadVariableOp:0(23training/Adam/conv2d_2/kernel/m/Initializer/zeros:0
�
training/Adam/conv2d_2/bias/m:0$training/Adam/conv2d_2/bias/m/Assign3training/Adam/conv2d_2/bias/m/Read/ReadVariableOp:0(21training/Adam/conv2d_2/bias/m/Initializer/zeros:0
�
training/Adam/dense/kernel/m:0#training/Adam/dense/kernel/m/Assign2training/Adam/dense/kernel/m/Read/ReadVariableOp:0(20training/Adam/dense/kernel/m/Initializer/zeros:0
�
training/Adam/dense/bias/m:0!training/Adam/dense/bias/m/Assign0training/Adam/dense/bias/m/Read/ReadVariableOp:0(2.training/Adam/dense/bias/m/Initializer/zeros:0
�
training/Adam/conv2d/kernel/v:0$training/Adam/conv2d/kernel/v/Assign3training/Adam/conv2d/kernel/v/Read/ReadVariableOp:0(21training/Adam/conv2d/kernel/v/Initializer/zeros:0
�
training/Adam/conv2d/bias/v:0"training/Adam/conv2d/bias/v/Assign1training/Adam/conv2d/bias/v/Read/ReadVariableOp:0(2/training/Adam/conv2d/bias/v/Initializer/zeros:0
�
!training/Adam/conv2d_1/kernel/v:0&training/Adam/conv2d_1/kernel/v/Assign5training/Adam/conv2d_1/kernel/v/Read/ReadVariableOp:0(23training/Adam/conv2d_1/kernel/v/Initializer/zeros:0
�
training/Adam/conv2d_1/bias/v:0$training/Adam/conv2d_1/bias/v/Assign3training/Adam/conv2d_1/bias/v/Read/ReadVariableOp:0(21training/Adam/conv2d_1/bias/v/Initializer/zeros:0
�
!training/Adam/conv2d_2/kernel/v:0&training/Adam/conv2d_2/kernel/v/Assign5training/Adam/conv2d_2/kernel/v/Read/ReadVariableOp:0(23training/Adam/conv2d_2/kernel/v/Initializer/zeros:0
�
training/Adam/conv2d_2/bias/v:0$training/Adam/conv2d_2/bias/v/Assign3training/Adam/conv2d_2/bias/v/Read/ReadVariableOp:0(21training/Adam/conv2d_2/bias/v/Initializer/zeros:0
�
training/Adam/dense/kernel/v:0#training/Adam/dense/kernel/v/Assign2training/Adam/dense/kernel/v/Read/ReadVariableOp:0(20training/Adam/dense/kernel/v/Initializer/zeros:0
�
training/Adam/dense/bias/v:0!training/Adam/dense/bias/v/Assign0training/Adam/dense/bias/v/Read/ReadVariableOp:0(2.training/Adam/dense/bias/v/Initializer/zeros:0
�
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
�
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
�
conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
s
conv2d_5/bias:0conv2d_5/bias/Assign#conv2d_5/bias/Read/ReadVariableOp:0(2!conv2d_5/bias/Initializer/zeros:08
�
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
�
training_2/Adam/iter:0training_2/Adam/iter/Assign*training_2/Adam/iter/Read/ReadVariableOp:0(2(training_2/Adam/iter/Initializer/zeros:0H
�
training_2/Adam/beta_1:0training_2/Adam/beta_1/Assign,training_2/Adam/beta_1/Read/ReadVariableOp:0(22training_2/Adam/beta_1/Initializer/initial_value:0H
�
training_2/Adam/beta_2:0training_2/Adam/beta_2/Assign,training_2/Adam/beta_2/Read/ReadVariableOp:0(22training_2/Adam/beta_2/Initializer/initial_value:0H
�
training_2/Adam/decay:0training_2/Adam/decay/Assign+training_2/Adam/decay/Read/ReadVariableOp:0(21training_2/Adam/decay/Initializer/initial_value:0H
�
training_2/Adam/learning_rate:0$training_2/Adam/learning_rate/Assign3training_2/Adam/learning_rate/Read/ReadVariableOp:0(29training_2/Adam/learning_rate/Initializer/initial_value:0H
�
#training_2/Adam/conv2d_3/kernel/m:0(training_2/Adam/conv2d_3/kernel/m/Assign7training_2/Adam/conv2d_3/kernel/m/Read/ReadVariableOp:0(25training_2/Adam/conv2d_3/kernel/m/Initializer/zeros:0
�
!training_2/Adam/conv2d_3/bias/m:0&training_2/Adam/conv2d_3/bias/m/Assign5training_2/Adam/conv2d_3/bias/m/Read/ReadVariableOp:0(23training_2/Adam/conv2d_3/bias/m/Initializer/zeros:0
�
#training_2/Adam/conv2d_4/kernel/m:0(training_2/Adam/conv2d_4/kernel/m/Assign7training_2/Adam/conv2d_4/kernel/m/Read/ReadVariableOp:0(25training_2/Adam/conv2d_4/kernel/m/Initializer/zeros:0
�
!training_2/Adam/conv2d_4/bias/m:0&training_2/Adam/conv2d_4/bias/m/Assign5training_2/Adam/conv2d_4/bias/m/Read/ReadVariableOp:0(23training_2/Adam/conv2d_4/bias/m/Initializer/zeros:0
�
#training_2/Adam/conv2d_5/kernel/m:0(training_2/Adam/conv2d_5/kernel/m/Assign7training_2/Adam/conv2d_5/kernel/m/Read/ReadVariableOp:0(25training_2/Adam/conv2d_5/kernel/m/Initializer/zeros:0
�
!training_2/Adam/conv2d_5/bias/m:0&training_2/Adam/conv2d_5/bias/m/Assign5training_2/Adam/conv2d_5/bias/m/Read/ReadVariableOp:0(23training_2/Adam/conv2d_5/bias/m/Initializer/zeros:0
�
"training_2/Adam/dense_1/kernel/m:0'training_2/Adam/dense_1/kernel/m/Assign6training_2/Adam/dense_1/kernel/m/Read/ReadVariableOp:0(24training_2/Adam/dense_1/kernel/m/Initializer/zeros:0
�
 training_2/Adam/dense_1/bias/m:0%training_2/Adam/dense_1/bias/m/Assign4training_2/Adam/dense_1/bias/m/Read/ReadVariableOp:0(22training_2/Adam/dense_1/bias/m/Initializer/zeros:0
�
#training_2/Adam/conv2d_3/kernel/v:0(training_2/Adam/conv2d_3/kernel/v/Assign7training_2/Adam/conv2d_3/kernel/v/Read/ReadVariableOp:0(25training_2/Adam/conv2d_3/kernel/v/Initializer/zeros:0
�
!training_2/Adam/conv2d_3/bias/v:0&training_2/Adam/conv2d_3/bias/v/Assign5training_2/Adam/conv2d_3/bias/v/Read/ReadVariableOp:0(23training_2/Adam/conv2d_3/bias/v/Initializer/zeros:0
�
#training_2/Adam/conv2d_4/kernel/v:0(training_2/Adam/conv2d_4/kernel/v/Assign7training_2/Adam/conv2d_4/kernel/v/Read/ReadVariableOp:0(25training_2/Adam/conv2d_4/kernel/v/Initializer/zeros:0
�
!training_2/Adam/conv2d_4/bias/v:0&training_2/Adam/conv2d_4/bias/v/Assign5training_2/Adam/conv2d_4/bias/v/Read/ReadVariableOp:0(23training_2/Adam/conv2d_4/bias/v/Initializer/zeros:0
�
#training_2/Adam/conv2d_5/kernel/v:0(training_2/Adam/conv2d_5/kernel/v/Assign7training_2/Adam/conv2d_5/kernel/v/Read/ReadVariableOp:0(25training_2/Adam/conv2d_5/kernel/v/Initializer/zeros:0
�
!training_2/Adam/conv2d_5/bias/v:0&training_2/Adam/conv2d_5/bias/v/Assign5training_2/Adam/conv2d_5/bias/v/Read/ReadVariableOp:0(23training_2/Adam/conv2d_5/bias/v/Initializer/zeros:0
�
"training_2/Adam/dense_1/kernel/v:0'training_2/Adam/dense_1/kernel/v/Assign6training_2/Adam/dense_1/kernel/v/Read/ReadVariableOp:0(24training_2/Adam/dense_1/kernel/v/Initializer/zeros:0
�
 training_2/Adam/dense_1/bias/v:0%training_2/Adam/dense_1/bias/v/Assign4training_2/Adam/dense_1/bias/v/Read/ReadVariableOp:0(22training_2/Adam/dense_1/bias/v/Initializer/zeros:0�`       ��2	Gz	����A*


epoch_lossBg?��D       `/�#	�z	����A*

	epoch_accAy6?l� N"       x=�	8{	����A*

epoch_val_lossd�?�-!       {��	s{	����A*

epoch_val_acc"�B?�3�