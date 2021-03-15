# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class EnableSchemaValidation:
    def __init__(self):
        pass

    def __enter__(self):
        from lale.settings import (
            disable_data_schema_validation,
            disable_hyperparams_schema_validation,
            set_disable_data_schema_validation,
            set_disable_hyperparams_schema_validation,
        )

        self.existing_data_schema_validation_flag = disable_data_schema_validation
        self.existing_hyperparams_schema_validation_flag = (
            disable_hyperparams_schema_validation
        )
        set_disable_data_schema_validation(False)
        set_disable_hyperparams_schema_validation(False)

    def __exit__(self, value, type, traceback):
        from lale.settings import (
            set_disable_data_schema_validation,
            set_disable_hyperparams_schema_validation,
        )

        set_disable_data_schema_validation(self.existing_data_schema_validation_flag)
        set_disable_hyperparams_schema_validation(
            self.existing_hyperparams_schema_validation_flag
        )
