import re
import matplotlib.pyplot as plt

filePath = './output/acc_loss_log.txt'

# Read data from the text file
with open(filePath, 'r') as file:
    lines = file.readlines()

# Initialize lists to store data
epoch = []
training_loss = []
validation_loss = []
training_mPJPE = []
validation_mPJPE = []
training_PAmPJPE = []
validation_PAmPJPE = []

# Parse data
for line in lines:
    parts = re.split(r',\s*|\s+', line.strip())
    # print(parts)
    epoch.append(int(parts[6]))
    training_loss.append(float(parts[10]))
    training_mPJPE.append(float(parts[14]))
    training_PAmPJPE.append(float(parts[16]))
    validation_loss.append(float(parts[21]))
    validation_mPJPE.append(float(parts[25]))
    validation_PAmPJPE.append(float(parts[27]))

# Plotting
plt.figure(figsize=(15, 10))
plt.plot(epoch, training_loss, label='Training Loss')
plt.plot(epoch, validation_loss, label='Validation Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('epoch_loss.png')
plt.close()

# Plot and save epoch vs mPJPE
plt.figure(figsize=(15, 10))
plt.plot(epoch, training_mPJPE, label='Training mPJPE')
plt.plot(epoch, validation_mPJPE, label='Validation mPJPE')
plt.title('Epoch vs mPJPE')
plt.xlabel('Epoch')
plt.ylabel('mPJPE')
plt.legend()
plt.tight_layout()
plt.savefig('epoch_mPJPE.png')
plt.close()

# Plot and save epoch vs PAmPJPE
plt.figure(figsize=(15, 10))
plt.plot(epoch, training_PAmPJPE, label='Training PAmPJPE')
plt.plot(epoch, validation_PAmPJPE, label='Validation PAmPJPE')
plt.title('Epoch vs PAmPJPE')
plt.xlabel('Epoch')
plt.ylabel('PAmPJPE')
plt.legend()
plt.tight_layout()
plt.savefig('epoch_PAmPJPE.png')
plt.close()