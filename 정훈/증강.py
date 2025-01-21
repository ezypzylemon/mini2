import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import os

# 데이터 로드
def load_all_data(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            all_data.append(data.reshape(data.shape[0], -1))
    return np.vstack(all_data)

# 생성자 모델
def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential([
        Input(shape=(latent_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    return model

# 판별자 모델
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
        # 데이터 준비
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_data = generator.predict(noise)

        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))

        # 판별자 학습
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)[0]
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)[0]
        discriminator.trainable = False

        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # 생성자 학습
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # 로그 출력
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

# 증강 데이터 생성 및 저장
def save_augmented_data(generator, latent_dim, num_samples, output_folder, output_dim):
    """
    증강 데이터를 생성하고 저장합니다.
    """
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    augmented_data = generator.predict(noise).reshape(-1, output_dim // 3, 3)

    os.makedirs(output_folder, exist_ok=True)
    for i, sample in enumerate(augmented_data):
        file_name = f"augmented_{i+1}.npy"
        file_path = os.path.join(output_folder, file_name)
        try:
            np.save(file_path, sample)
            print(f"[SUCCESS] Saved: {file_path}")
        except Exception as e:
            print(f"[FAILED] Could not save {file_path}: {e}")

# 실행
if __name__ == "__main__":
    folder_path = "C:/Users/Admin/Desktop/motion_data/좌표"
    augmented_output_folder = "C:/Users/Admin/Desktop/motion_data/gan_augmented_data"
    os.makedirs(augmented_output_folder, exist_ok=True)

    # 데이터 로드
    data = load_all_data(folder_path)
    latent_dim = 100
    output_dim = data.shape[1]

    # 모델 정의
    generator = build_generator(latent_dim, output_dim)
    discriminator = build_discriminator(output_dim)

    # 판별자 컴파일
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 디버깅: 컴파일 확인
    print("Discriminator compiled:", discriminator.loss is not None)

    # GAN 모델 정의
    discriminator.trainable = False
    noise_input = layers.Input(shape=(latent_dim,))
    generated_output = generator(noise_input)
    validity = discriminator(generated_output)

    gan = Model(noise_input, validity)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # 디버깅: 모델 요약 출력
    print("Generator summary:")
    generator.summary()
    print("\nDiscriminator summary:")
    discriminator.summary()
    print("\nGAN summary:")
    gan.summary()

    # GAN 학습
    train_gan(generator, discriminator, gan, data, latent_dim, epochs=1000, batch_size=32)

    # 증강 데이터 생성 및 저장
    num_augmented_samples = 100
    save_augmented_data(generator, latent_dim, num_augmented_samples, augmented_output_folder, output_dim)
