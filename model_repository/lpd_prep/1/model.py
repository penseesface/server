
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
            model_config, "LPD_output1_prep")


        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])


    def execute(self, requests):

        # def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        #     # Resize and pad image while meeting stride-multiple constraints
        #     h,w = img.shape[:2]  # current shape [height, width]

        #     l = h if h > w else w

        #     scale = new_shape[0]/l

        #     h,w = np.array(img.shape[:2])*scale
            
        #     resized = cv2.resize(img,(int(w),int(h)))
            
        #     dh = max(new_shape[0]-resized.shape[0],0)
        #     dw = max(new_shape[1]-resized.shape[1],0)

        #     img = cv2.copyMakeBorder(resized, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)  # add border

        #     return img #, scale



        output_dtype = self.output0_dtype
        responses = []
        for request in requests:
            input_ = pb_utils.get_input_tensor_by_name(request, "LPD_input1_prep")
            unq_image = np.squeeze(input_.as_numpy())
            rgb_image = cv2.cvtColor(unq_image, cv2.COLOR_BGR2RGB)

            #img = letterbox(rgb_image)

            img=rgb_image.transpose(2, 0, 1).astype(float)  # BGR to RGB,
            
            imgs = np.float32(img)/255

            out = imgs.astype(dtype=np.float32)

            output = pb_utils.Tensor("LPD_output1_prep", out.astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
