#!/bin/bash

#Remember to set the location of the github folder.
github_dir="{path_to_github_dir}"

#NetTCR 2.2 (NetTCR-2.2)
python $github_dir/src/nettcr_predict.py -i $github_dir/data/test.csv -g $github_dir -m $github_dir/models/NetTCR_2_3/paired/limited/15 -tb $github_dir/models/TCRbase/paired/limited/IMMREP_pred_df.csv -o $github_dir/predictions/NetTCR_2_2_predictions.csv -a 10

#NetTCR 2.3 - Mixed (M1)
python $github_dir/src/nettcr_predict_ensemble.py -i $github_dir/data/test.csv -g $github_dir -m $github_dir/data/model_ensemble_files/M1.txt -tb $github_dir/models/TCRbase/alpha_beta/IMMREP_pred_df.csv -o $github_dir/predictions/NetTCR_M1_predictions.csv -a 0

#NetTCR 2.3 - Mixed_Limited (M2)
python $github_dir/src/nettcr_predict_ensemble.py -i $github_dir/data/test.csv -g $github_dir -m $github_dir/data/model_ensemble_files/M2_v2.txt -tb $github_dir/models/TCRbase/alpha_beta/ensemble_limited/IMMREP_pred_df.csv -o $github_dir/predictions/NetTCR_M2_v2_predictions.csv -a 0

#NetTCR 2.3 - Mixed_Limited_TCRbase (M3)
python $github_dir/src/nettcr_predict_ensemble.py -i $github_dir/data/test.csv -g $github_dir -m $github_dir/data/model_ensemble_files/M3_v2.txt -tb $github_dir/models/TCRbase/alpha_beta/ensemble_limited/IMMREP_pred_df.csv -o $github_dir/predictions/NetTCR_M3_v2_predictions.csv -a 10

#NetTCR 2.3 - ABP_Limited_TCRbase (M4)
python $github_dir/src/nettcr_predict_ensemble.py -i $github_dir/data/test.csv -g $github_dir -m $github_dir/data/model_ensemble_files/M4_v2.txt -tb $github_dir/models/TCRbase/alpha_beta/ensemble_limited/IMMREP_pred_df.csv -o $github_dir/predictions/NetTCR_M4_v2_predictions.csv -a 10

#NetTCR 2.3 - Ensemble_Limited_TCRbase (M5)
python $github_dir/src/nettcr_predict_ensemble.py -i $github_dir/data/test.csv -g $github_dir -m $github_dir/data/model_ensemble_files/M5_v2.txt -tb $github_dir/models/TCRbase/alpha_beta/ensemble_limited/IMMREP_pred_df.csv -o $github_dir/predictions/NetTCR_M5_v2_predictions.csv -a 10

#NetTCR 2.2 - ensemble_bootstrapping (NetTCR-2.2-b)
python $github_dir/src/nettcr_predict.py -i $github_dir/data/test.csv -g $github_dir -m $github_dir/models/NetTCR_2_3/paired/limited/ensemble_bootstrapping -tb $github_dir/models/TCRbase/paired/limited/IMMREP_pred_df.csv -o $github_dir/predictions/NetTCR_2_2_b_predictions.csv -a 10
