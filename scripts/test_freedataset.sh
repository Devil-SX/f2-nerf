mkdir logs -p
conda activate f2-nerf
python scripts/run.py --config-name=free dataset_name=free_dataset mode=train +work_dir=$(pwd) case_name=grass      | tee -a logs/free_dataset.log
python scripts/run.py --config-name=free dataset_name=free_dataset mode=train +work_dir=$(pwd) case_name=hydrant    | tee -a logs/free_dataset.log
python scripts/run.py --config-name=free dataset_name=free_dataset mode=train +work_dir=$(pwd) case_name=lab    | tee -a logs/free_dataset.log
python scripts/run.py --config-name=free dataset_name=free_dataset mode=train +work_dir=$(pwd) case_name=pillar | tee -a logs/free_dataset.log
python scripts/run.py --config-name=free dataset_name=free_dataset mode=train +work_dir=$(pwd) case_name=road   | tee -a logs/free_dataset.log
python scripts/run.py --config-name=free dataset_name=free_dataset mode=train +work_dir=$(pwd) case_name=sky    | tee -a logs/free_dataset.log
python scripts/run.py --config-name=free dataset_name=free_dataset mode=train +work_dir=$(pwd) case_name=stair  | tee -a logs/free_dataset.log