# Word-Level GPT-2 Implementation for Better Text Generation
# This builds a word-level version that produces coherent sentences

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import sys
import os
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')

def safe_print(text):
    """Safely print text that might contain Unicode characters"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

print("TensorFlow version:", tf.__version__)

# Step 1: Prepare Sample Text Data
print("\n=== STEP 1: PREPARING TEXT DATA ===")

# Expanded sample text - much larger for better training
sample_text = """
The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully under the tree.
The fox was very clever and agile. It ran through the forest with great speed.
The forest was full of tall trees and beautiful flowers. Birds were singing in the trees.
The sun was shining brightly through the leaves. It was a perfect day for adventure.
Many animals lived in the forest including rabbits, squirrels, and deer.
The rabbit was white and fluffy. The squirrel collected nuts for winter.
The deer gracefully moved through the meadow. All animals were friends in this magical place.
Every morning the sun would rise and paint the sky with golden colors.
The wind would gently blow through the grass making a soft whistling sound.
Children would often visit this forest to play and explore nature.
They would climb trees and pick berries. The forest was their playground.
Stories were told around campfires about the magical creatures that lived there.
Some said there were fairies that danced in the moonlight.
Others believed that wise old owls gave advice to lost travelers.
The forest had many secrets waiting to be discovered by curious minds.
Adventure awaited those brave enough to explore the unknown paths.
Magic filled the air when the moon was full and bright.
Travelers would follow the stars to find their way home.
The forest whispered ancient secrets to those who listened carefully.
Wisdom could be found in the rustling of leaves and the flow of streams.
The old wizard lived in a tower made of stone and magic.
He spent his days reading ancient books and practicing spells.
The villagers would come to him for advice and healing.
His knowledge of herbs and potions was legendary throughout the land.
The young apprentice learned the ways of magic from the wise master.
Together they protected the village from dark forces and evil creatures.
The castle stood on a hill overlooking the peaceful valley below.
Knights in shining armor patrolled the walls and protected the kingdom.
The princess was kind and beautiful with golden hair and bright blue eyes.
She loved to read books and explore the castle gardens.
The dragon lived in a cave deep in the mountains far away.
It was said to guard a treasure of gold and precious jewels.
Brave knights would attempt to slay the dragon but few returned.
The dragon was not evil but simply protecting its ancient home.
The kingdom was ruled by a wise and just king who cared for his people.
He made fair laws and ensured everyone had food and shelter.
The queen was graceful and elegant with a heart full of compassion.
She would often visit the poor and sick to offer comfort and aid.
The royal family was loved by all the people in the kingdom.
They lived in harmony with nature and respected all living things.
The ancient library contained thousands of books filled with knowledge.
Scholars from far and wide would travel to study the ancient texts.
The librarian was a wise old woman who knew every book by heart.
She would help visitors find exactly what they were looking for.
The marketplace was bustling with merchants selling their wares.
Farmers brought fresh fruits and vegetables from their fields.
Artisans crafted beautiful pottery and woven cloth for sale.
The blacksmith made strong tools and weapons for the villagers.
The baker filled the air with the sweet smell of fresh bread.
The butcher provided meat and fish for the hungry families.
The tailor sewed fine clothes for the wealthy and simple garments for the poor.
The cobbler made sturdy shoes for walking on rough roads.
The miller ground grain into flour for making bread and cakes.
The weaver created beautiful fabrics from wool and cotton.
The potter shaped clay into useful vessels and decorative items.
The carpenter built houses and furniture for the growing village.
The mason carved stone into strong walls and beautiful buildings.
The painter decorated walls with colorful murals and designs.
The musician played sweet melodies on his wooden flute.
The dancer moved gracefully to the rhythm of the music.
The storyteller captivated audiences with tales of adventure and magic.
The jester made everyone laugh with his clever jokes and tricks.
The guard kept watch over the village gates day and night.
The messenger carried important letters between distant towns.
The shepherd tended his flock of sheep in the green meadows.
The fisherman cast his net into the clear blue lake.
The hunter tracked deer and rabbits in the dense forest.
The gatherer collected berries and herbs from the wild places.
The healer used natural remedies to cure illness and injury.
The midwife helped bring new life into the world.
The teacher shared knowledge with eager young minds.
The student studied hard to learn new skills and wisdom.
The friend offered comfort and support in times of need.
The neighbor shared food and tools with those who had less.
The stranger was welcomed with kindness and hospitality.
The traveler brought news and stories from distant lands.
The merchant traded goods and brought prosperity to the village.
The craftsman took pride in creating useful and beautiful things.
The artist expressed beauty and emotion through their work.
The writer captured thoughts and feelings in words on paper.
The poet found rhythm and meaning in the sounds of language.
The philosopher pondered deep questions about life and existence.
The scientist observed the world and discovered its secrets.
The explorer ventured into unknown territories and mapped new lands.
The inventor created new tools and machines to make life easier.
The engineer designed bridges and buildings that stood the test of time.
The architect planned beautiful and functional spaces for people to live and work.
The gardener tended plants and created beautiful landscapes.
The farmer worked the land to grow food for everyone.
The cook prepared delicious meals that brought people together.
The host welcomed guests with warmth and generosity.
The guest showed gratitude and respect for their generous hosts.
The family shared love and support through good times and bad.
The community worked together to solve problems and celebrate successes.
The village grew and prospered through the hard work of its people.
The kingdom flourished under wise and caring leadership.
The world was full of wonder and beauty waiting to be discovered.
"""

print(f"Text length: {len(sample_text)} characters")

# Step 2: Word-Level Text Preprocessing
print("\n=== STEP 2: WORD-LEVEL TEXT PREPROCESSING ===")

def preprocess_text_words(text):
    """Clean and prepare text for word-level training"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Tokenize into words
    words = word_tokenize(text)
    
    # Filter out punctuation and keep only words
    words = [word for word in words if word.isalpha()]
    
    # Remove very short words (likely noise)
    words = [word for word in words if len(word) > 1]
    
    return words

# Preprocess the text
words = preprocess_text_words(sample_text)
print(f"Number of words: {len(words)}")
print(f"Sample words: {words[:20]}")

# Step 3: Create Word-Level Vocabulary
print("\n=== STEP 3: CREATING WORD VOCABULARY ===")

# Get unique words and their frequencies
word_counts = Counter(words)
vocab_size = min(1000, len(word_counts))  # Limit vocabulary size

# Get most common words
most_common_words = word_counts.most_common(vocab_size)
vocab_words = [word for word, count in most_common_words]

# Add special tokens
vocab_words = ['<PAD>', '<UNK>', '<START>', '<END>'] + vocab_words
vocab_size = len(vocab_words)

print(f"Vocabulary size: {vocab_size}")
print(f"Sample vocabulary: {vocab_words[:20]}")

# Create word mappings
word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
idx_to_word = {idx: word for idx, word in enumerate(vocab_words)}

# Convert words to numbers
words_as_int = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
print(f"Words as numbers length: {len(words_as_int)}")

# Step 4: Create Training Sequences
print("\n=== STEP 4: CREATING TRAINING SEQUENCES ===")

# Sequence length - how many words to look at to predict the next one
seq_length = 10  # Shorter for word-level (10 words vs 80 characters)
examples_per_epoch = len(words_as_int) // (seq_length + 1)

# Create training examples
def create_training_examples_words(words, seq_length):
    """Create input-output pairs for word-level training"""
    input_text = []
    target_text = []
    
    for i in range(0, len(words) - seq_length, 1):
        input_seq = words[i:i + seq_length]
        target_seq = words[i + 1:i + seq_length + 1]
        
        input_text.append(input_seq)
        target_text.append(target_seq)
    
    return np.array(input_text), np.array(target_text)

input_text, target_text = create_training_examples_words(words_as_int, seq_length)
print(f"Training examples: {len(input_text)}")
print(f"Input shape: {input_text.shape}")
print(f"Target shape: {target_text.shape}")

# Show example
example_idx = 0
print(f"\nExample {example_idx}:")
print(f"Input:  {' '.join([idx_to_word[idx] for idx in input_text[example_idx]])}")
print(f"Target: {' '.join([idx_to_word[idx] for idx in target_text[example_idx]])}")

# Step 5: Build the Word-Level Model
print("\n=== STEP 5: BUILDING THE WORD-LEVEL MODEL ===")

def build_word_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """Build a word-level model"""
    model = keras.Sequential([
        # Embedding layer - converts word indices to vectors
        keras.layers.Embedding(vocab_size, embedding_dim),
        
        # LSTM layer - captures sequential patterns
        keras.layers.LSTM(rnn_units, 
                         return_sequences=True, 
                         stateful=True,
                         recurrent_initializer='glorot_uniform'),
        
        # Dropout for regularization
        keras.layers.Dropout(0.3),
        
        # Another LSTM layer for more complexity
        keras.layers.LSTM(rnn_units, 
                         return_sequences=True, 
                         stateful=True,
                         recurrent_initializer='glorot_uniform'),
        
        # Dropout for regularization
        keras.layers.Dropout(0.3),
        
        # Output layer - predicts next word
        keras.layers.Dense(vocab_size)
    ])
    
    return model

# Model parameters - optimized for word-level
embedding_dim = 128
rnn_units = 256
batch_size = 16

# Build the model
model = build_word_model(vocab_size, embedding_dim, rnn_units, batch_size)

# Build the model with proper input shape for stateful LSTM
model.build(input_shape=(batch_size, None))
print("Model created successfully!")

# Step 6: Compile and Train the Model
print("\n=== STEP 6: COMPILING AND TRAINING ===")

# Loss function for word prediction
def loss_function(real, pred):
    """Custom loss function for word prediction"""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

# Compile the model
model.compile(
    optimizer='adam',
    loss=loss_function,
    metrics=['accuracy']
)

# Show model summary
model.summary()

# Prepare data for training
dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True)

print("Starting training...")
history = model.fit(
    dataset,
    epochs=30,   # Fewer epochs needed for word-level
    verbose=1
)

# Step 7: Create Model for Text Generation
print("\n=== STEP 7: CREATING GENERATION MODEL ===")

# Build a model for generation (batch_size=1)
generation_model = build_word_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# Build the generation model with proper input shape first
generation_model.build(input_shape=(1, None))

# Copy the trained weights
generation_model.set_weights(model.get_weights())

print("Generation model ready!")

# Step 8: Word-Level Text Generation Function
print("\n=== STEP 8: CREATING WORD-LEVEL GENERATION FUNCTION ===")

def reset_lstm_states(model):
    """Reset states for all LSTM layers in a Sequential model."""
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()

def generate_text_words(model, start_string, num_generate=20, temperature=1.0):
    """Generate text using the trained word-level model"""
    # Tokenize the start string
    start_words = word_tokenize(start_string.lower())
    start_words = [word for word in start_words if word.isalpha() and len(word) > 1]
    
    # Convert to indices
    input_eval = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in start_words]
    
    # Pad or truncate to seq_length
    if len(input_eval) < seq_length:
        input_eval = [word_to_idx['<PAD>']] * (seq_length - len(input_eval)) + input_eval
    else:
        input_eval = input_eval[-seq_length:]
    
    input_eval = tf.expand_dims(input_eval, 0)
    
    # Empty list to store our results
    text_generated = []
    
    # Reset states
    reset_lstm_states(model)
    
    for i in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        
        # Using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        # Pass the predicted word as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)
        
        # Convert back to word
        predicted_word = idx_to_word[predicted_id]
        if predicted_word not in ['<PAD>', '<UNK>', '<START>', '<END>']:
            text_generated.append(predicted_word)
    
    return start_string + ' ' + ' '.join(text_generated)

# Step 9: Test Word-Level Text Generation
print("\n=== STEP 9: TESTING WORD-LEVEL TEXT GENERATION ===")

# Generate text with different starting phrases
test_phrases = [
    "the quick",
    "the forest",
    "children would",
    "the sun",
    "magic"
]

print("Generated text samples:")
print("=" * 60)

for phrase in test_phrases:
    generated_text = generate_text_words(
        generation_model, 
        start_string=phrase,
        num_generate=15,
        temperature=0.7
    )
    
    safe_print(f"\nStarting with: '{phrase}'")
    safe_print("-" * 40)
    safe_print(generated_text)
    safe_print("")

# Step 10: Plot Training History
print("\n=== STEP 10: ANALYZING TRAINING RESULTS ===")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Step 11: Interactive Word-Level Generation
print("\n=== STEP 11: INTERACTIVE WORD-LEVEL GENERATION ===")

def interactive_generation_words():
    """Interactive word-level text generation"""
    print("Interactive Word-Level Text Generation (type 'quit' to exit)")
    while True:
        start_text = input("\nEnter starting text: ").strip()
        if start_text.lower() == 'quit':
            break
        if len(start_text) == 0:
            start_text = "the"
        try:
            reset_lstm_states(generation_model)
            generated = generate_text_words(
                generation_model,
                start_string=start_text,
                num_generate=20,
                temperature=0.7
            )
            safe_print(f"\nGenerated text:\n{generated}")
        except Exception as e:
            print(f"Error generating text: {e}")

print("\nSuccess! Your word-level GPT-2 model is trained and ready!")
print("You can now use interactive_generation_words() to generate text!")

print("\n=== WHAT YOU'VE BUILT ===")
print("✓ A word-level language model that produces coherent sentences")
print("✓ Real word generation with proper vocabulary")
print("✓ Customizable creativity with temperature control")
print("✓ Interactive word-level text generation")

print("\n=== TO USE WITH YOUR OWN TEXT ===")
print("1. Replace the sample_text variable with your own text")
print("2. Or load from a file: with open('your_file.txt', 'r') as f: sample_text = f.read()")
print("3. The model will learn word patterns in your text!")

print("\n=== EXPERIMENT WITH ===")
print("- Different temperature values (0.5 = conservative, 1.0 = creative)")
print("- Longer training (increase epochs)")
print("- Different starting phrases")
print("- Larger vocabulary size for more diverse words") 