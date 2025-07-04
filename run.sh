#!/bin/bash

#SBATCH -J proc              # Job name
#SBATCH -o .out/job_%a.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=40

#SBATCH --array=0-61
##SBATCH --array=0-25

if [[ $1 == "power" ]]
then
    python -O save_power.py $SLURM_ARRAY_TASK_ID 1 1  "cue" $2 1
    python -O save_power.py $SLURM_ARRAY_TASK_ID 2 0  "cue" $2 1
    # python -O save_power.py $SLURM_ARRAY_TASK_ID 1 1  "cue" $2 20
    # python -O save_power.py $SLURM_ARRAY_TASK_ID 2 0  "cue" $2 20
fi


if [[ $1 == "pec" ]]
then
	python -O power_to_power_cc.py $SLURM_ARRAY_TASK_ID 1 1 0 "cue" $2 
fi


if [[ $1 == "pec2" ]]
then
	python power_events_coincidence.py $SLURM_ARRAY_TASK_ID 1 1 0 "cue" $2 50
	python power_events_coincidence.py $SLURM_ARRAY_TASK_ID 1 1 0 "cue" $2 80
	python power_events_coincidence.py $SLURM_ARRAY_TASK_ID 1 1 0 "cue" $2 95
fi

# if [[ $1 == "coherence" ]]
# then
    #python -O save_coherences.py "pec" "cue" $SLURM_ARRAY_TASK_ID 0 1000 "lucy"
    # python -O save_coherences.py "coh" "cue" $SLURM_ARRAY_TASK_ID 1 1000 $2 
# fi

#if [[ $1 == "cohthr" ]]
#then
#    python -O compute_coherence_thresholds.py "pec" "lucy"
#fi

if [[ $1 == "ratemod" ]]
then
    # python -O rate_modulations.py $SLURM_ARRAY_TASK_ID $2 "cue" 0 20
    #python -O rate_modulations.py $SLURM_ARRAY_TASK_ID $2 "cue" 3 20
    #python -O rate_modulations.py $SLURM_ARRAY_TASK_ID $2 "cue" .70
    #python -O rate_modulations.py $SLURM_ARRAY_TASK_ID $2 "cue" .90
    # python -O rate_modulations.py $SLURM_ARRAY_TASK_ID $2 "cue" 0. 1
    # python -O rate_modulations.py $SLURM_ARRAY_TASK_ID $2 "cue" .95 1
    python -O rate_modulations.py $SLURM_ARRAY_TASK_ID $2 "cue" 0. 1
fi

if [[ $1 == "degree" ]]
then
    python -O save_network_analysis.py "coh" $SLURM_ARRAY_TASK_ID "cue" $2 0
fi

if [[ $1 == "powerenc" ]]
then
    #python -O mi_power_analysis.py "zpow" 1 1 "cue" 1 $2 0 # No SLVR
    # python -O mi_power_analysis.py "pow" 1 1 0 "cue" 1 $2 0 # No SLVR
    # python -O mi_power_analysis.py "pow" 1 1 80 "cue" 1 $2 0 # No SLVR
    # python -O mi_power_analysis.py "pow" 1 1 0 "cue" 1 $2 0 # No SLVR
    python -O mi_power_analysis.py "pow" 1 1 50 "cue" 1 $2 0 # No SLVR
    python -O mi_power_analysis.py "pow" 1 1 80 "cue" 1 $2 0 # No SLVR
    python -O mi_power_analysis.py "pow" 1 1 95 "cue" 1 $2 0 # No SLVR
    # python -O mi_power_analysis.py "pow" 1 1 95 "cue" 1 $2 0 # No SLVR
#    python -O mi_power_analysis.py "pow" 1 1 0 "cue" 1 $2 1 # No SLVR
#    python -O mi_power_analysis.py "pow" 1 1 80 "cue" 1 $2 1 # No SLVR
		 
fi

#if [[ $1 == "crkenc" ]]
#then
#    # python -O mi_crackle_analysis.py 1 1 .7 "cue" $2  
#    python -O mi_crackle_analysis.py 1 1 .8 "cue" $2 
#    # python -O mi_crackle_analysis.py 1 1 .9 "cue" $2 
#fi
#
#if [[ $1 == "cohenc" ]]
#then
#    #python -O mi_coh_analysis.py "coh" 1 $2 "cue" 
#    python -O mi_coh_analysis.py "pec" 1 $2 "cue" 
#fi
#
#if [[ $1 == "cocrkenc" ]]
#then
#    python -O mi_cocrk_analysis.py 1 1 "cue" 1 $2 1 0 
#fi

if [[ $1 == "crkstats" ]]
then
   #python -O crackle_stats.py $SLURM_ARRAY_TASK_ID $2 .7
    #python -O crackle_stats.py $SLURM_ARRAY_TASK_ID $2 .8
    python -O crackle_stats.py $SLURM_ARRAY_TASK_ID $2 0 0 
    python -O crackle_stats.py $SLURM_ARRAY_TASK_ID $2 0 1
    python -O crackle_stats.py $SLURM_ARRAY_TASK_ID $2 0 2
fi

if [[ $1 == "avalanche" ]]
then
		python -O temporal_components.py $SLURM_ARRAY_TASK_ID 50 $2 0 1 1 "relative" 1
		python -O temporal_components.py $SLURM_ARRAY_TASK_ID 80 $2 0 1 1 "relative" 1
		python -O temporal_components.py $SLURM_ARRAY_TASK_ID 95 $2 0 1 1 "relative" 1
		#python -O temporal_components.py $SLURM_ARRAY_TASK_ID 80 $2 0 1 1 "relative" 1
		#python -O temporal_components.py $SLURM_ARRAY_TASK_ID 90 $2 0 1 1 "relative" 1
		#python -O temporal_components.py $SLURM_ARRAY_TASK_ID 95 $2 0 1 1 "relative" 1
    # correct task
    #python -O temporal_components.py $SLURM_ARRAY_TASK_ID 85 $2 0 1 1 "absolute" 5
    #python -O temporal_components.py $SLURM_ARRAY_TASK_ID 90 $2 0 1 1 "absolute" 5
    #python -O temporal_components.py $SLURM_ARRAY_TASK_ID 95 $2 0 1 1 "absolute" 5
fi
