import matplotlib.pyplot as plt
import pandas as pd
just_attention = pd.read_csv('just_attenton.csv', delimiter=',').values
with_beam = pd.read_csv('beam.csv', delimiter=',').values
adaptive = pd.read_csv('adaptive_train.csv', delimiter=',').values

print(just_attention.shape)
print(with_beam.shape)

plt.plot(just_attention[:,0], just_attention[:,1], marker='X', label="just_attention_token_accuracy")
plt.plot(with_beam[:,0], with_beam[:,1], marker='X', label="attention_and_beamSearch_token_accuracy")
plt.plot(just_attention[:,0], just_attention[:,2], marker='^', label="just_attention_denotation_accuracy")
plt.plot(with_beam[:,0], with_beam[:,2], marker='^', label="attention_and_beamSearch_denotation_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Dev Accuracy")
plt.legend()
plt.savefig("beam.png")

plt.clf()

plt.plot(adaptive[:,0], adaptive[:,1], marker='X', label="adaptive_training_token_accuracy")
plt.plot(with_beam[:,0], with_beam[:,1], marker='X', label="regular_training_token_accuracy")
plt.plot(adaptive[:,0], adaptive[:,2], marker='^', label="adaptive_training_denotation_accuracy")
plt.plot(with_beam[:,0], with_beam[:,2], marker='^', label="regular_training_denotation_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Dev Accuracy")
plt.legend()
plt.savefig("adaptive.png")
