python load_searchr1_get_RAG_dragin_score.py --dataset truthfulqa --model_path /disk/Yitong/models/v05_step_200
python load_searchr1_get_RAG_dragin_score.py --dataset truthfulqa --model_path /disk/Yitong/models/v05_step_100


python load_searchr1_get_RAG_dragin_score.py --dataset halueval --model_path /disk/Yitong/models/v05_step_200
python load_searchr1_get_RAG_dragin_score.py --dataset halueval --model_path /disk/Yitong/models/v05_step_100



python run_DRAGIN.py --dataset halueval
python run_DRAGIN.py --dataset truthfulqa


python run_FLARE.py --dataset halueval
python run_FLARE.py --dataset truthfulqa


python run_R1Searcher.py --dataset halueval
python run_R1Searcher.py --dataset truthfulqa

python run_Searcho1.py --dataset halueval
python run_Searcho1.py --dataset truthfulqa