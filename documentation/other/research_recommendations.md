##


## Culturally Grounded Augmentation Strategies

To achieve a dataset of "thousands" of recordings that reflect the polyphonic complexity of film soundtracks, synthetic augmentation is required. Simple random mixing of monophonic clips is a common technique to simulate polyphonic textures, but it often lacks musical coherence. Improved strategies involve "domain-aware mixing," where instruments are combined based on shared modal structures, such as Chinese pentatonic modes or Persian dastgāhs, and aligned tempo (BPM). This approach ensures that the synthesized tracks closely mimic authentic performances, where instruments follow both the same tonal center and a coherent rhythmic pace.

For the specific context of 1960s films, augmentation must also include synthetic degradation. Adding modeled tape hiss, hum, and non-linear distortion ensures the model generalizes to the "poor quality" signals found in archival film repositories. Furthermore, the inclusion of "musical interference" through pitch-shifting and time-stretching helps the model distinguish between overlapping instruments in dense action sequences.

## Technical Objective II: Data Preprocessing and DSP Pipelines

The second technical objective involves developing data preprocessing pipelines that apply digital signal processing (DSP) to generate reproducible spectrograms. In the context of martial arts films, the choice of time-frequency representation is critical for distinguishing between the sharp transients of foley effects and the sustained harmonics of traditional instruments.

### Feature Extraction and Fusion

Standard MIR approaches often rely on the Mel-spectrogram, which maps the frequency spectrum to the Mel scale to approximate human perception. However, musical signals, particularly those from instruments with rich overtones like the Pipa or Guzheng, benefit from the Constant-Q Transform (CQT). The CQT provides log-frequency resolution where frequency bins are geometrically spaced, keeping the distance between musical notes constant across the spectrum.

Recent research in Chinese instrument classification suggests that stacking multiple features, MFCC, CQT, and Chroma, can capture diverse sound information more effectively than a single input. Multi-channel input strategies allow the model to process these distinct features in parallel. For example, a 3-channel spectrogram feature stacking method, combined with hybrid channel-spatial attention mechanisms, has demonstrated test accuracies as high as 98.79% in identifying traditional instruments.

### Audio Restoration and Source Isolation

Archival soundtracks from the 1960s are typically monophonic and already mixed, creating a significant challenge for instrument identification. Music Source Restoration (MSR) extends traditional source separation to include dereverberation, bandwidth extension, and declipping. Techniques such as spectral inpainting, which function similarly to "Content-Aware Fill" in image editing, can reconstruct lost audio data by analyzing the surrounding texture and tonal patterns.

Once restored, the audio is segmented into uniform time windows for processing. Research indicates that 5-second segments are optimal for Chinese instruments; shorter windows can lead to information loss, while longer windows can create sparse data that complicates model training.

## Technical Objective III: Architectural Design and Model Exploration

The third technical objective focuses on designing, training, and testing deep learning models, particularly convolutional neural networks (CNNs), on martial arts film soundtracks. The evolution of these models has seen a transition from simple classification to sophisticated sequence-to-sequence transcription.

### Convolutional and Recurrent Foundations

CNNs are the backbone of most contemporary instrument identification systems due to their ability to learn abstract spectral characteristics. Models such as ResNet-50 and Inception-V3 have been adapted via transfer learning to recognize instruments in polyphonic signals, often outperforming traditional machine learning methods like Support Vector Machines (SVM) by nearly 20% in $F_{1}$ measure. For martial arts films, CNNs treat the spectrogram as an image, using position-based kernels to obtain local information about the timbral signature of an instrument.

However, CNNs are inherently limited in capturing long-term temporal dependencies, which are crucial for detecting note onsets and offsets. This has led to the adoption of Convolutional Recurrent Neural Networks (CRNNs), which combine CNN feature extraction with Recurrent Neural Network (RNN) layers like LSTM or GRU. CRNNs are particularly effective for "predominant" instrument recognition in real-world polyphonic music.

### The "Onsets and Frames" Paradigm

A major breakthrough in polyphonic music transcription occurred with the "Onsets and Frames" model architecture. This system is designed to jointly predict onsets, the start of a note, and frame-wise pitches. A critical innovation of this architecture is the reduction of false positives; it ensures that a note pitch is not predicted as active unless a corresponding onset has been detected.

In follow-up studies, this paradigm was expanded into Multi-Labeled Note States Classification (MLNSC) systems capable of predicting four distinct states: onsets, offsets, velocities, and frame-wise pitches. While successful, these frame-level models often have limited resolution due to the "hop size" of the sampling process. To address this, High-Resolution Time Regression (HRTR) systems use algorithms to determine the precise continuous timing of onsets and offsets by regressing the distance to the nearest event rather than performing binary classification.

### Transformer-Based Sequence-to-Sequence Models

The current state-of-the-art in multi-instrument transcription involves Transformer architectures that treat the task as a sequence-to-sequence problem. Models such as MT3 (Multi-Task Multitrack Music Transcription) utilize a MIDI-like decoding transformer to generate note event tokens. These models replace traditional piano-roll outputs with tokens representing pitch, onset, offset, and instrument class.

The YourMT3+ framework, an enhancement of the MT3 model, introduces hierarchical attention transformers and cross-dataset stem augmentation. The use of spectral cross-attention (SCA) allows the model to focus on the relationship between input audio and output representations, leveraging the "sparse" nature of acoustic-to-symbolic correspondence. This is particularly useful for martial arts films where a specific instrument activation may only correspond to a narrow time window in the audio.

| Model Architecture | Key Mechanism | Best Use Case | Reference |
| --- | --- | --- | --- |
| ResNet / CNN | Local spectral filters | Predominant instrument classification |  |
| CRNN (CNN+LSTM) | Spectral-temporal fusion | Handling complex polyphonic textures |  |
| Onsets and Frames | Joint prediction path | High-precision transcription |  |
| MT3 / YourMT3+ | Sequence-to-sequence tokens | Multi-instrument archival search |  |
| MERT (SSL) | Masked Language Modeling | Foundation for transfer learning |  |

## Advanced Learning Paradigms: SSL and Transfer Learning

To address the research gaps in identifying rare or traditional instruments where labeled data is scarce, this project leverages pre-trained foundation models and self-supervised learning (SSL).

### MERT: Foundation Model for Music Understanding

MERT (Music undERstanding model with large-scale self-supervised Training) represents a paradigm shift in MIR. It is a transformer-based model pre-trained on approximately 60,000 hours of music using a masked language modeling (MLM) objective. During pre-training, the model learns to predict discrete tokens of masked audio parts, using a neural codec such as Encodec as a tokenizer and an auxiliary CQT reconstruction loss to capture acoustic information.

MERT achieves state-of-the-art performance in numerous MIR tasks, including instrument classification and pitch detection, despite using less than 2% of the parameters of larger models like Jukebox. By using MERT as a base, the martial arts film project can leverage deep representations of pitch and timbre that are already generalized across diverse musical styles.

### Fine-Tuning for Chinese Traditional Instruments

The "target domain" for this project, 1960s Chinese film music, is significantly different from the general music datasets used to train models like MERT. Transfer learning allows knowledge from the "source domain," general music, to be transferred to the target domain. Fine-tuning a pre-trained model on datasets like ChMusic or CTIS allows it to specialize in the unique timbral qualities of Chinese instruments, such as the Erhu's "soft and delicate" tone.

Research into Persian instrument recognition has demonstrated that fine-tuning a MERT model with culturally grounded data augmentation can achieve state-of-the-art multi-instrument classification accuracy even in complex polyphonic settings. A similar approach is recommended for this project: starting with a pre-trained MERT encoder and adding a multi-label classification head for instrument recognition, alongside a transformer decoder for onset/offset transcription.

## Overcoming Technical Hurdles in Polyphonic Localization

The objective of determining when an instrument is played, onsets and offsets, introduces specific challenges regarding temporal precision and class imbalance.

### High-Resolution Onset and Offset Detection

In the context of film soundtracks, where instruments may be masked by explosive sound effects, the "Acoustic-Transformer" model is a promising architecture. This model consists of a CNN-based acoustic model to estimate frame-wise pitch probabilities and a Transformer-based language model to evaluate the global correlation between pitch combinations. This structure enables the model to concurrently obtain local information for onsets and global context for sustaining notes.

The precision of these estimates is measured by the Overlap Ratio and the F1 score within a narrow tolerance window. For onsets, a tolerance of $\pm 50\text{ ms}$ is standard. For offsets, the criteria are more stringent, requiring the predicted end-time to be within 20% of the reference duration or 50 ms. Sequence-to-sequence approaches simplify this by integrating onset, offset, and velocity prediction into a single generative process, effectively modeling the temporal dependencies between frames that are crucial for accurate music transcription.

### Addressing Unbalanced Datasets with Focal Loss

Archival film data inevitably contains unbalanced instrument quantities due to the prevalence of certain instruments in 1960s orchestration. Standard binary cross-entropy (BCE) loss can be overwhelmed by "easy negatives," the silence or absence of an instrument, causing the model to neglect rare instruments.

To overcome this, Weighted Focal Loss (W-FL) is proposed. Focal Loss introduces a modulating factor $(1 - p_{t})^{\gamma}$ to the standard cross-entropy loss, where $p_{t}$ is the model's estimated probability for the correct class. The factor $\gamma$ adjusts the rate at which easy examples are down-weighted, forcing the model to pay more attention to "difficult" samples such as rare instruments or instruments masked by noise.

The Focal Loss function is defined as:

$$
FL(p_{t}) = -\alpha_{t} (1 - p_{t})^{\gamma} \log(p_{t})
$$

where $\gamma = 2$ is frequently found to have the best performance in MIR tasks. This approach is particularly effective for multi-label classification where each frame can contain multiple instruments.

## Evaluation Frameworks for Music Information Retrieval

Evaluating the performance of an automated system on martial arts soundtracks requires metrics that account for both instrument identification (classification) and temporal accuracy (transcription).

### Identification Metrics

The F1-score is the standard metric, defined as the harmonic mean of precision and recall. In polyphonic contexts, both Micro-averaged and Macro-averaged metrics must be reported. Micro-averaging treats every instance equally, making it sensitive to class imbalance, while macro-averaging treats every instrument category equally, providing a better measure of the system's ability to recognize rare instruments.

$$
\text{Precision}_{micro} = \frac{\sum_{l=1}^{L} TP_{l}}{\sum_{l=1}^{L} (TP_{l} + FP_{l})} \quad \text{Recall}_{micro} = \frac{\sum_{l=1}^{L} TP_{l}}{\sum_{l=1}^{L} (TP_{l} + FN_{l})}
$$

### Transcription Metrics

For the localization of onsets and offsets, the Label Ranking Average Precision (LRAP) is used to assess the ranking of predicted instruments. Additionally, `mir_eval` provides standard heuristics for computing precision, recall, and overlap ratios specifically for note-level transcription. Systems that achieve high precision in these metrics enable researchers to perform "content-based audio search," where an archive can be queried for segments where a Pipa and Erhu are playing simultaneously.

## Future Outlook: Practical Applications and Cultural Preservation

Developing an automated system for 1960s martial arts cinema has profound implications for cultural preservation and music education. The ability to automatically annotate instrument activations reduces the burden of manual labeling, making vast film archives searchable by instrumental cues. This technology can accelerate soundtrack selection in traditional music education and provide researchers with quantitative tools to analyze the aesthetic evolution of the Shaw Brothers' "new wuxia" era.

Furthermore, the integration of these models with optical music recognition (OMR) pipelines could lead to the automatic generation of scores from historic audio, preserving performance techniques that were previously only transmitted orally. As hardware advances, the potential for implementing even more complex model architectures, such as Mamba for efficient long-context transcription, will further enhance our ability to protect and study this rich intangible cultural heritage.

## Practical Recommendations for System Development

To meet the project's technical objectives effectively, the following recommendations are proposed:

- **Architecture:** Adopt a hybrid Acoustic-Transformer or YourMT3+ framework. This leverages a CNN encoder for robust local feature extraction from 1960s audio while utilizing a Transformer decoder to handle the sequence-to-sequence nature of onset/offset transcription.
- **Fine-Tuning:** Utilize MERT as the pre-trained foundation model. Fine-tuning MERT on a combination of CTIS and Western orchestral datasets will provide the model with the necessary timbral cross-context to navigate mixed soundtracks.
- **Preprocessing:** Use 3-channel feature stacking (MFCC, CQT, Chroma) as the input to the CNN encoder. This fusion captures the rhythmic, harmonic, and timbral characteristics of traditional instruments more accurately than single-feature inputs.
- **Loss Function:** Implement Weighted Focal Loss to ensure the model does not become biased toward the more common orchestral strings and can identify the subtle cues of rare Chinese folk instruments even in dense mixes.
- **Augmentation:** Apply culturally grounded random mixing that respects musical modes and tempo to generate the necessary thousands of training samples from isolated recordings.
