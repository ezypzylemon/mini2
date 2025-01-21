import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# 데이터 로드
def load_all_data(folder_path):
    """
    폴더 내의 모든 .npy 파일을 로드하여 하나의 배열로 합칩니다.
    Args:
        folder_path (str): .npy 파일이 저장된 폴더 경로.
    Returns:
        ndarray: (num_samples, num_joints * 3) 형태의 데이터.
    """
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)  # (num_samples, num_joints, 3)
            all_data.append(data.reshape(data.shape[0], -1))  # (num_samples, num_joints * 3)
    return np.vstack(all_data)  # 모든 데이터를 하나로 합침

# 생성자 모델 (Generator)
def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential([
        Input(shape=(latent_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    return model

# 판별자 모델 (Discriminator)
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN 모델 학습
def train_gan(generator, discriminator, gan, data, latent_dim, epochs=1000, batch_size=32):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # 판별자 학습
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_data = generator.predict(noise)

        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # 생성자 학습
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # 학습 과정 출력
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

# 데이터 증강
def generate_augmented_data(generator, num_samples, latent_dim, output_dim):
    """
    증강 데이터를 생성합니다.
    Args:
        generator (Model): 학습된 생성자 모델.
        num_samples (int): 생성할 증강 데이터 수.
        latent_dim (int): 잠재 공간 크기.
        output_dim (int): 출력 데이터 크기.
    Returns:
        ndarray: (num_samples, num_joints, 3) 형태의 증강 데이터.
    """
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise)
    return generated_data.reshape(-1, output_dim // 3, 3)  # Reshape to (num_samples, num_joints, 3)

# 실행 예제
if __name__ == "__main__":
    folder_path = "C:/Users/Admin/Desktop/motion_data/좌표"  # .npy 파일이 있는 폴더 경로
    augmented_output_folder = "C:/Users/Admin/Desktop/motion_data/gan_augmented_data"
    os.makedirs(augmented_output_folder, exist_ok=True)

    data = load_all_data(folder_path)
    latent_dim = 100  # 잠재 공간 차원
    output_dim = data.shape[1]  # num_joints * 3

    generator = build_generator(latent_dim, output_dim)
    discriminator = build_discriminator(output_dim)

    # GAN 모델
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    noise_input = layers.Input(shape=(latent_dim,))
    generated_output = generator(noise_input)
    validity = discriminator(generated_output)
    gan = Model(noise_input, validity)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # 학습
    train_gan(generator, discriminator, gan, data, latent_dim, epochs=1000, batch_size=32)

    # 증강 데이터 생성
    num_augmented_samples = 100
    augmented_data = generate_augmented_data(generator, num_augmented_samples, latent_dim, output_dim)

    # 증강 데이터 저장
    for i, sample in enumerate(augmented_data):
        output_path = os.path.join(augmented_output_folder, f"augmented_{i+1}.npy")
        np.save(output_path, sample)
        print(f"Saved augmented data to: {output_path}")
