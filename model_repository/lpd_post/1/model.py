
import numpy as np
import sys
import json
import cv2

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "LPD_output1_post")


        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])


    def execute(self, requests):

        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            y = np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
            y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
            y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
            return y

        def nms(boxes, scores, iou_thres):

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            scores = scores
        
            areas = (x2 - x1 + 1) * (y2 - y1 + 1) #所有box面积
            order = scores.argsort()[::-1] #降序排列得到scores的坐标索引
        
            keep = []
            while order.size > 0:
                i = order[0] #最大得分box的坐标索引
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]]) 
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]]) #最高得分的boax与其他box的公共部分(交集)
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1) #求高和宽，并使数值合法化
                inter = w * h #其他所有box的面积
                ovr = inter / (areas[i] + areas[order[1:]] - inter)  #IOU:交并比
                inds = np.where(ovr <= iou_thres)[0] #ovr小表示两个box交集少，可能是另一个物体的框，故需要保留
                order = order[inds + 1]  #iou小于阈值的框


            return keep

        def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=10):
            """Runs Non-Maximum Suppression (NMS) on inference results

            Returns:
                list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            """

            nc = prediction.shape[2] - 5  # number of classes

            xc = prediction[..., 4] > conf_thres  # candidates

            # Checks
            assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
            assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

            # Settings
            min_wh, max_wh = 20, 640  # (pixels) minimum and maximum box width and height
            max_nms = 300  # maximum number of boxes into torchvision.ops.nms()

            output = [np.zeros((0, 6))] * prediction.shape[0]
            for xi, x in enumerate(prediction):  # image index, image inference

                x = x[xc[xi]]  # confidence

                # If none remain process next image
                if not x.shape[0]:
                    continue

                # Compute conf
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf


                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box = xywh2xyxy(x[:, :4])


                # Detections matrix nx6 (xyxy, conf, cls)
                conf  = x[:, 5:].max(1, keepdims=True)


                j  = x[:, 5:].argmax(1).astype(np.float32).reshape(-1,1)

                x = np.concatenate((box, conf, j), 1)

                # Check shape
                n = x.shape[0]  # number of boxes
                if not n:  # no boxes
                    continue
                elif n > max_nms:  # excess boxes
                    x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

                # Batched NMS
                c = x[:, 5:6] * max_wh  # classes
                boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

                keep = nms(boxes, scores, iou_thres)  # NMS

                left = len(keep)

                if left>int(max_det):
                    keep = keep[:max_det]

                output[xi] = x[keep]
                


            return output




        output_dtype = self.output0_dtype
        responses = []
        for request in requests:
            input_ = pb_utils.get_input_tensor_by_name(request, "LPD_input1_post")
            
            input_ = np.array([input_.as_numpy()])


            pred = non_max_suppression(input_)
            #print(pred[0].shape)

            dets=[]
            for i, det in enumerate(pred):  # detections per image

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size


                    l_w = det[:,2] - det[:,0]
                    l_y = det[:,3] - det[:,1]

                    expand = 0.15
                    #expand = 0

                    det[:,0] = np.clip(det[:,0]-l_w*expand,0,640)
                    det[:,1] = np.clip(det[:,1]-l_y*expand,0,640)
                    det[:,2] = np.clip(det[:,2]+l_w*expand,0,640)
                    det[:,3] = np.clip(det[:,3]+l_y*expand,0,640)

                    dets.append(det)
            if len(dets)>0:
                dets = np.array([dets[0][0]])
            else: 
                dets = np.zeros((1,6))

            output = pb_utils.Tensor("LPD_output1_post", dets.astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
