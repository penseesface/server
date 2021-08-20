
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
            model_config, "LPR_output1_post")


        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])


    def execute(self, requests):

        def LPR_postprocess(output):
            myLP = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
            final = ''	
            #final = []	
            str_idxes = []
            prev=100
            for i,x in enumerate(output[0]):
                if i == 0 :
                    prev = x
                    str_idxes.append(x)
                else:
                    if x != prev:
                        str_idxes.append(x)
                    prev = x

                #str_idxes.append(x)
            for i in str_idxes:
                if int(i) != 35:
                    final += myLP[int(i)]
                    #final.append(myLP[int(i)])

            return final

        output_dtype = self.output0_dtype
        responses = []
        for request in requests:
            input_ = pb_utils.get_input_tensor_by_name(request, "LPR_input1_post")
            input_ = np.array([input_.as_numpy()])
            
            det = pb_utils.get_input_tensor_by_name(request, "LPR_input2_post")
            det = np.array([det.as_numpy()])


            r = LPR_postprocess(input_)
            print(r)
            fianl_return = [r]
            for e in det[0][0]:
                fianl_return.append(str(e))

            output = pb_utils.Tensor("LPR_output1_post", np.array(fianl_return).astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
