/**
  ******************************************************************************
  * @file    maintenance_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-03-24T22:07:23+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef MAINTENANCE_DATA_PARAMS_H
#define MAINTENANCE_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_MAINTENANCE_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_maintenance_data_weights_params[1]))
*/

#define AI_MAINTENANCE_DATA_CONFIG               (NULL)


#define AI_MAINTENANCE_DATA_ACTIVATIONS_SIZES \
  { 2560, }
#define AI_MAINTENANCE_DATA_ACTIVATIONS_SIZE     (2560)
#define AI_MAINTENANCE_DATA_ACTIVATIONS_COUNT    (1)
#define AI_MAINTENANCE_DATA_ACTIVATION_1_SIZE    (2560)



#define AI_MAINTENANCE_DATA_WEIGHTS_SIZES \
  { 316948, }
#define AI_MAINTENANCE_DATA_WEIGHTS_SIZE         (316948)
#define AI_MAINTENANCE_DATA_WEIGHTS_COUNT        (1)
#define AI_MAINTENANCE_DATA_WEIGHT_1_SIZE        (316948)



#define AI_MAINTENANCE_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_maintenance_activations_table[1])

extern ai_handle g_maintenance_activations_table[1 + 2];



#define AI_MAINTENANCE_DATA_WEIGHTS_TABLE_GET() \
  (&g_maintenance_weights_table[1])

extern ai_handle g_maintenance_weights_table[1 + 2];


#endif    /* MAINTENANCE_DATA_PARAMS_H */
