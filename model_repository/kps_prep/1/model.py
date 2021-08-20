
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
            model_config, "KPS_output1_prep")


        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])


    def execute(self, requests):

        output_dtype = self.output0_dtype
        responses = []
        for request in requests:
            raw_img = pb_utils.get_input_tensor_by_name(request, "KPS_input1_prep")
            raw_img = np.squeeze(raw_img.as_numpy())

            det = pb_utils.get_input_tensor_by_name(request, "KPS_input2_prep")
            det = np.squeeze(det.as_numpy())

            raw_img=raw_img.transpose(1, 2, 0).astype(float) 
            
            # for x1,y1,x2,y2,conf,cl in det:
            x1,y1,x2,y2,conf,cl = det
            img_rgb = np.zeros((3,64,64))
            if conf > 0.35:
                crop = raw_img[int(y1):int(y2),int(x1):int(x2)]
                img_rgb  = cv2.resize(crop,(64,64))

                img_rgb = np.transpose(img_rgb,(2,1,0))

            output = pb_utils.Tensor("KPS_output1_prep", img_rgb.astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
