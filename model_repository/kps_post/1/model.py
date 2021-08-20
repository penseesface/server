
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
            model_config, "KPS_output1_post")


        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])


    def execute(self, requests):

        def LP_align(crop,double_line,pts):

            w, h = size_single = 289, 80
            ref_single = np.matrix([[0, w, w, 0],[0, 0, h, h]],np.float32).T

            w, h = size_double = 185, 142
            ref_double = np.matrix([[0, w, w, 0],[0, 0, h, h]],np.float32).T


            if double_line : 
                ref = ref_double
                size = size_double
            else:
                ref = ref_single
                size = size_single

            h_mtx = cv2.getPerspectiveTransform(pts, ref)
            crop = cv2.warpPerspective(crop, h_mtx, size, flags=cv2.INTER_CUBIC)

            if double_line:
                h, w, _ = crop.shape
                new_w = w//5
                up = crop[:h//2, new_w:new_w*4, :]
                new_h = up.shape[0]
                bo = crop[h//2:(h//2+new_h), w//40:, :]
                new_crop = np.concatenate((up, bo), axis=1)
            else:
                new_crop = crop

            return new_crop


        output_dtype = self.output0_dtype
        responses = []
        for request in requests:
            kps_pts = pb_utils.get_input_tensor_by_name(request, "KPS_input1_post")
            kps_pts = np.squeeze(kps_pts.as_numpy())

            kps_index = pb_utils.get_input_tensor_by_name(request, "KPS_input2_post")
            kps_index = np.squeeze(kps_index.as_numpy())
           
            raw_img = pb_utils.get_input_tensor_by_name(request, "KPS_input3_post")
            raw_img = np.squeeze(raw_img.as_numpy())

            det = pb_utils.get_input_tensor_by_name(request, "KPS_input4_post")
            det = np.squeeze(det.as_numpy())

            raw_img=raw_img.transpose(1, 2, 0).astype(float) 
            
            # for x1,y1,x2,y2,conf,cl in det:
            x1,y1,x2,y2,conf,cl = det

            lp_aligned = np.zeros((3,48,96))
            if conf > 0.3:

                crop = raw_img[int(y1):int(y2),int(x1):int(x2)]

                c_h, c_w, _ = crop.shape
                KPS_ratio_w = 64/c_w
                KPS_ratio_h = 64/c_h

                kps_index=kps_index.repeat(8)
                kps_pts = kps_pts[:,kps_index][:,0]
                kps_pts = kps_pts.reshape(2, 4).T # 4xy
                kps_pts[:,0] /=KPS_ratio_w
                kps_pts[:,1] /=KPS_ratio_h
                kps_pts[:,0] = np.clip(kps_pts[:,0],0,c_w)
                kps_pts[:,1] = np.clip(kps_pts[:,1],0,c_h)

                lp_aligned = LP_align(crop,int(cl),kps_pts)

                lp_resized = cv2.resize(lp_aligned,(96,48))

                lp_aligned = np.transpose(lp_resized,(2,0,1))

            output = pb_utils.Tensor("KPS_output1_post", lp_aligned.astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
