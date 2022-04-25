
# Config parameters for KT module

data_path='./data/train-all-2.pkl'
n_question=13523
max_seq=200
n_pid=-1
n_blocks=1
d_model=256
dropout=0.05
kq_same=1
n_heads=8
d_ff=2048
l2=1e-5
final_fc_dim=512
batch_size=36
learning_rate=1e-5
max_learning_rate=2e-3
n_epochs=12

# Save the model
base_path='./models'
model_name='kt_model'


# Training
train-kt:
	python -m train \
		--data_path $(data_path) \
		--n_question $(n_question) \
		--max_seq $(max_seq) \
		--n_pid $(n_pid) \
		--n_blocks $(n_blocks) \
		--d_model $(d_model) \
		--dropout $(dropout) \
		--kq_same $(kq_same) \
		--n_heads $(n_heads) \
		--d_ff $(d_ff) \
		--l2 $(l2) \
		--final_fc_dim $(final_fc_dim) \
		--batch_size $(batch_size) \
		--learning_rate $(learning_rate) \
		--max_learning_rate $(max_learning_rate) \
		--n_epochs $(n_epochs) \
		--base_path $(base_path) \
		--model_name $(model_name)


# Config parameters for KCCP module
module_type='kccp'
data_path='./data/train-all-subset.pkl'
kn_concept=188
kinput_dim=100
knum_layers=2
khidden_dim=256
kdropout=0.2
kmax_seq=200

kbatch_size=48
klearning_rate=0.001
kmax_learning_rate=2e-2
kn_epochs=20

# Save the model
base_path='./models'
kccp_model_name='kccp_model'

train-kccp:
	python -m train \
		--module_type $(module_type) \
		--data_path $(data_path) \
		--n_concept $(kn_concept) \
		--max_seq $(kmax_seq) \
		--input_dim $(kinput_dim) \
		--num_layers $(knum_layers) \
		--hidden_dim $(khidden_dim) \
		--dropout $(kdropout) \
		--batch_size $(kbatch_size) \
		--learning_rate $(klearning_rate) \
		--max_learning_rate $(kmax_learning_rate) \
		--n_epochs $(kn_epochs) \
		--base_path $(base_path) \
		--model_name $(kccp_model_name)