from typing import List, Union

class KwargsParser:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def parameter_in_kwargs(self, parameter_name):
        return parameter_name in self.kwargs

    def get_parameter(self, parameter_name, exptected_type, default_value):
        pass

    def get_parameters(self, parameter_names: List[str], expected_types: List[Union[type, List[type]]]):
        pass

    def check_parameters_do_not_cooccur(self, list_a, list_b):
        for variable_a in list_a:
            if self.parameter_in_kwargs(variable_a):
                for variable_b in list_b:
                    if self.parameter_in_kwargs(variable_b):
                        return False
        return True

