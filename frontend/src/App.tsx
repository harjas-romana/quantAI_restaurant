import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import gsap from 'gsap'
import './App.css'

const prompts = [
  'Get fresh perspectives on tricky hospital problems',
  'Brainstorm healthcare ideas',
  'Rewrite message for maximum impact',
  'Summarize key medical points',
];

const API_BASE = 'http://localhost:8000'; // Updated

// Server Status Type
type ServerStatus = {
  status: 'online' | 'offline' | 'loading';
  version: string;
  lastChecked: Date;
};

export default function App() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Kia ora! Welcome to QuantAI Hospital. How can I assist you today?' },
  ]);
  const [input, setInput] = useState('');
  const [voiceMode, setVoiceMode] = useState(false);
  const [listening, setListening] = useState(false);
  const [splash, setSplash] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [languages, setLanguages] = useState<string[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [showLangDropdown, setShowLangDropdown] = useState(false);
  const [serverStatus, setServerStatus] = useState<ServerStatus>({
    status: 'loading',
    version: '',
    lastChecked: new Date()
  });
  const parallaxRefs = [
    useRef<HTMLDivElement>(null),
    useRef<HTMLDivElement>(null),
    useRef<HTMLDivElement>(null),
    useRef<HTMLDivElement>(null),
    useRef<HTMLDivElement>(null),
    useRef<HTMLDivElement>(null),
  ];
  const recognitionRef = useRef<any>(null);
  const gradientTextRef = useRef<HTMLDivElement>(null);

  // Add a mapping for language logos/icons (move inside component for JSX scope)
  const languageIcons: Record<string, React.ReactNode> = {
    english: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#22d3ee" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">EN</text></svg>
    ),
    spanish: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#f87171" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">ES</text></svg>
    ),
    french: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#60a5fa" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">FR</text></svg>
    ),
    german: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#fbbf24" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">DE</text></svg>
    ),
    hindi: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#a3e635" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">HI</text></svg>
    ),
    chinese: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#f59e42" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">中</text></svg>
    ),
    japanese: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#fbbf24" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">日</text></svg>
    ),
    italian: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#34d399" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">IT</text></svg>
    ),
    russian: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#818cf8" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">RU</text></svg>
    ),
    arabic: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#fbbf24" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">ع</text></svg>
    ),
    portuguese: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#10b981" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">PT</text></svg>
    ),
    korean: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#f472b6" /><text x="12" y="16" textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">한</text></svg>
    ),
    // Add more as needed
  };

  // Check server health status
  useEffect(() => {
    const checkServerHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        setServerStatus({
          status: 'online',
          version: data.version || '1.0.0',
          lastChecked: new Date()
        });
      } catch (e) {
        setServerStatus(prev => ({
          ...prev,
          status: 'offline',
          lastChecked: new Date()
        }));
      }
    };

    checkServerHealth();
    const interval = setInterval(checkServerHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Fetch languages for dropdown
  useEffect(() => {
    const fetchLanguages = async () => {
      try {
        const res = await fetch(`${API_BASE}/languages`);
        if (!res.ok) throw new Error('Failed to fetch languages');
        const data = await res.json();
        if (data.languages && Array.isArray(data.languages)) {
          setLanguages(data.languages);
        }
      } catch (error) {
        console.error('Error fetching languages:', error);
        setError('Failed to load languages');
      }
    };
    fetchLanguages();
  }, []);

  // Moving gradient animation
  useEffect(() => {
    if (gradientTextRef.current) {
      gsap.to(gradientTextRef.current, {
        backgroundPosition: '200% center',
        duration: 10,
        repeat: -1,
        ease: 'none'
      });
    }
  }, []);

  // Enhanced splash screen effect with staggered animations
  useEffect(() => {
    const timer = setTimeout(() => setSplash(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  // 3D Parallax effect with GSAP and floating elements
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const { innerWidth, innerHeight } = window;
      const x = (e.clientX / innerWidth - 0.5);
      const y = (e.clientY / innerHeight - 0.5);
      gsap.to(parallaxRefs[0].current, { x: x * 40, y: y * 40, duration: 0.7, ease: 'power2.out' });
      gsap.to(parallaxRefs[1].current, { x: x * 80, y: y * 80, duration: 0.9, ease: 'power2.out' });
      gsap.to(parallaxRefs[2].current, { x: x * 120, y: y * 120, duration: 1.1, ease: 'power2.out' });
      gsap.to(parallaxRefs[3].current, { x: x * 160, y: y * 160, duration: 1.3, ease: 'power2.out' });
      gsap.to(parallaxRefs[4].current, { x: x * 200, y: y * 200, duration: 1.5, ease: 'power2.out' });
      gsap.to(parallaxRefs[5].current, { x: x * 240, y: y * 240, duration: 1.7, ease: 'power2.out' });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [parallaxRefs]);

  // Web Speech API for speech-to-text
  const startListening = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      setError('Speech recognition not supported in this browser.');
      return;
    }
    setError(null);
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = selectedLanguage || 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      setListening(false);
    };
    recognition.onerror = (event: any) => {
      setListening(false);
      setError('Could not understand your voice. Please try again.');
    };
    recognition.onend = () => setListening(false);
    recognition.start();
    setListening(true);
    recognitionRef.current = recognition;
  };

  // Web Speech API for text-to-speech
  const speak = (text: string) => {
    if (!('speechSynthesis' in window)) return;
    const utter = new window.SpeechSynthesisUtterance(text);
    utter.lang = selectedLanguage || 'en-US';
    window.speechSynthesis.speak(utter);
  };

  // ElevenLabs text-to-speech
  const [isSpeaking, setIsSpeaking] = useState<string | null>(null);
  
  const speakWithElevenLabs = async (text: string, messageId: number) => {
    try {
      setIsSpeaking(messageId.toString());
      const res = await fetch(`${API_BASE}/text-to-speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      
      if (!res.ok) throw new Error('TTS server error');
      
      const data = await res.json();
      if (data.success && data.audio_url) {
        // Play the audio
        const audio = new Audio(`${API_BASE}${data.audio_url}`);
        audio.play();
        
        // Reset speaking state when audio finishes playing
        audio.onended = () => {
          setIsSpeaking(null);
        };
      }
    } catch (e) {
      console.error('ElevenLabs TTS error:', e);
      setError('Error generating speech. Falling back to browser TTS.');
      // Fallback to browser TTS
      speak(text);
      setIsSpeaking(null);
    } finally {
      // Don't reset here, we'll reset when audio finishes playing
    }
  };

  // Send text to backend
  const sendText = async (text: string) => {
    setMessages((msgs) => [...msgs, { sender: 'user', text }]);
    setInput('');
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/text-query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: selectedLanguage }),
      });
      if (!res.ok) throw new Error('Server error');
      const data = await res.json();
      setMessages((msgs) => [...msgs, { sender: 'bot', text: data.response }]);
      // We're not auto-speaking anymore since we have a dedicated button for ElevenLabs TTS
    } catch (e) {
      setError('Sorry, there was an error connecting to QuantAI Hospital.');
      setMessages((msgs) => [...msgs, { sender: 'bot', text: 'Sorry, there was an error.' }]);
    } finally {
      setLoading(false);
    }
  };

  // Voice recording (for voice mode)
  const handleVoiceInput = async () => {
    if (!('MediaRecorder' in window)) {
      setError('Voice recording not supported in this browser.');
      return;
    }
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000
        } 
      });
      
      // Use audio/wav directly if supported
      const mimeType = MediaRecorder.isTypeSupported('audio/wav') 
        ? 'audio/wav' 
        : 'audio/webm';
      
      const recorder = new MediaRecorder(stream, {
        mimeType: mimeType,
        audioBitsPerSecond: 16000
      });
      
      const chunks: BlobPart[] = [];
      
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        try {
          const audioBlob = new Blob(chunks, { type: mimeType });
          
          // Always convert to WAV format to ensure compatibility
          const wavBlob = await convertAudioToWav(audioBlob);
          console.log('Audio converted to WAV format');
          
          await sendVoice(wavBlob);
          stream.getTracks().forEach(track => track.stop());
        } catch (error) {
          console.error('Error processing audio:', error);
          setError('Error processing audio. Please try again.');
          setListening(false);
        }
      };
      
      recorder.start();
      setListening(true);
      
      // Stop recording after 4 seconds
      setTimeout(() => {
        if (recorder.state === 'recording') {
          recorder.stop();
          setListening(false);
        }
      }, 4000);
      
    } catch (e) {
      console.error('Microphone error:', e);
      setError('Error accessing microphone. Please check permissions.');
      setListening(false);
    }
  };

  // Convert any audio format to WAV
  const convertAudioToWav = async (audioBlob: Blob): Promise<Blob> => {
    return new Promise(async (resolve, reject) => {
      try {
        // Create an audio context
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
          sampleRate: 16000
        });
        
        // Read the audio data
        const arrayBuffer = await audioBlob.arrayBuffer();
        
        // Decode the audio data
        audioContext.decodeAudioData(arrayBuffer, (audioBuffer) => {
          // Get the raw PCM data
          const numberOfChannels = 1; // Mono
          const sampleRate = 16000;
          const length = audioBuffer.length;
          
          // Create WAV header
          const wavHeader = createWavHeader(length, numberOfChannels, sampleRate, 16);
          
          // Get audio data
          const channelData = audioBuffer.getChannelData(0);
          const samples = new Int16Array(length);
          
          // Convert float32 to int16
          for (let i = 0; i < length; i++) {
            const s = Math.max(-1, Math.min(1, channelData[i]));
            samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          
          // Combine header and samples
          const wavBytes = new Uint8Array(wavHeader.length + samples.byteLength);
          wavBytes.set(wavHeader, 0);
          wavBytes.set(new Uint8Array(samples.buffer), wavHeader.length);
          
          // Create WAV blob
          const wavBlob = new Blob([wavBytes], { type: 'audio/wav' });
          resolve(wavBlob);
        }, (err) => {
          console.error('Error decoding audio:', err);
          reject(err);
        });
      } catch (error) {
        console.error('Error converting audio:', error);
        reject(error);
      }
    });
  };
  
  // Create a proper WAV header
  const createWavHeader = (dataLength: number, numChannels: number, sampleRate: number, bitsPerSample: number): Uint8Array => {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    
    // RIFF identifier
    writeString(view, 0, 'RIFF');
    // File length
    view.setUint32(4, 36 + dataLength * 2, true);
    // WAVE identifier
    writeString(view, 8, 'WAVE');
    // Format chunk marker
    writeString(view, 12, 'fmt ');
    // Format chunk length
    view.setUint32(16, 16, true);
    // Sample format (1 is PCM)
    view.setUint16(20, 1, true);
    // Channel count
    view.setUint16(22, numChannels, true);
    // Sample rate
    view.setUint32(24, sampleRate, true);
    // Byte rate (sample rate * block align)
    view.setUint32(28, sampleRate * numChannels * bitsPerSample / 8, true);
    // Block align (channel count * bytes per sample)
    view.setUint16(32, numChannels * bitsPerSample / 8, true);
    // Bits per sample
    view.setUint16(34, bitsPerSample, true);
    // Data chunk marker
    writeString(view, 36, 'data');
    // Data chunk length
    view.setUint32(40, dataLength * 2, true);
    
    return new Uint8Array(header);
  };

  // Helper function to write strings to DataView
  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  // Send voice to backend
  const sendVoice = async (blob: Blob) => {
    setMessages((msgs) => [...msgs, { sender: 'user', text: '[Voice message sent]' }]);
    setInput('');
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('audio_file', blob, 'voice.wav');
    
    try {
      console.log('Sending voice data to server...');
      const res = await fetch(`${API_BASE}/voice-query`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: 'Server error' }));
        console.error('Voice query error:', data);
        if (data.detail && data.detail.includes('Could not understand audio input')) {
          setError('Sorry, I could not understand your voice. Please try again or speak more clearly.');
          setMessages((msgs) => [...msgs, { sender: 'bot', text: 'Sorry, I could not understand your voice. Please try again or speak more clearly.' }]);
        } else {
          throw new Error('Server error');
        }
        return;
      }
      
      const data = await res.json();
      console.log('Voice response received:', data);
      setMessages((msgs) => [...msgs, { sender: 'bot', text: data.text }]);
      
      // We'll let the user click the speak button to hear the response
      // This gives them control over when to play audio
      
    } catch (e) {
      console.error('Voice query error:', e);
      setError('Sorry, there was an error connecting to QuantAI Hospital.');
      setMessages((msgs) => [...msgs, { sender: 'bot', text: 'Sorry, there was an error.' }]);
    } finally {
      setLoading(false);
    }
  };

  // Handle form submit
  const handleSubmit = async (e: any) => {
    e.preventDefault();
    if (!input.trim()) return;
    if (voiceMode && input.trim()) {
      await sendText(input);
    } else {
      await sendText(input);
    }
  };

  // Floating animated elements for extra polish
  const floatingShapes = [
    { className: 'bg-green-200', size: 32, top: '10%', left: '5%', delay: 0 },
    { className: 'bg-green-300', size: 20, top: '80%', left: '80%', delay: 0.5 },
    { className: 'bg-teal-200', size: 16, top: '60%', left: '20%', delay: 1 },
    { className: 'bg-green-100', size: 24, top: '30%', left: '90%', delay: 1.5 },
    { className: 'bg-teal-100', size: 14, top: '75%', left: '10%', delay: 2 },
  ];

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-green-50 to-white relative overflow-hidden">
      {/* Advanced 3D Splash Screen with Parallax */}
      <AnimatePresence>
        {splash && (
          <motion.div
            className="fixed inset-0 z-50 flex flex-col items-center justify-center overflow-hidden perspective-1000"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1 }}
          >
            {/* Animated Background Elements */}
            <div className="absolute inset-0 bg-gradient-to-br from-green-50 via-white to-blue-50 z-0" />
            
            {/* Parallax Background Elements */}
            <motion.div 
              className="absolute inset-0 z-0"
              animate={{ 
                backgroundPosition: ['0% 0%', '100% 100%'],
              }}
              transition={{ 
                duration: 20, 
                ease: "linear", 
                repeat: Infinity, 
                repeatType: "reverse" 
              }}
              style={{ 
                backgroundImage: 'radial-gradient(circle at center, rgba(167, 243, 208, 0.2) 0%, transparent 50%)',
                backgroundSize: '100% 100%',
              }}
            />
            
            {/* 3D Floating Elements */}
            {[...Array(12)].map((_, i) => (
              <motion.div
                key={`particle-${i}`}
                className="absolute rounded-full bg-gradient-to-r from-green-200 to-teal-100 shadow-xl"
                style={{ 
                  width: 20 + Math.random() * 40,
                  height: 20 + Math.random() * 40,
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  opacity: 0.4 + Math.random() * 0.3,
                  filter: 'blur(2px)',
                  zIndex: 1
                }}
                animate={{
                  x: [
                    Math.random() * 100 - 50,
                    Math.random() * 100 - 50,
                    Math.random() * 100 - 50
                  ],
                  y: [
                    Math.random() * 100 - 50,
                    Math.random() * 100 - 50,
                    Math.random() * 100 - 50
                  ],
                  scale: [1, 1.1, 0.9, 1],
                  rotate: [0, 180, 360],
                }}
                transition={{
                  duration: 15 + Math.random() * 15,
                  repeat: Infinity,
                  repeatType: "reverse",
                  ease: "easeInOut"
                }}
              />
            ))}
            
            {/* Main Content with Parallax Effect */}
            <motion.div
              className="relative z-10 flex flex-col items-center"
              animate={{ y: [0, -10, 0] }}
              transition={{ 
                duration: 6, 
                repeat: Infinity, 
                ease: "easeInOut" 
              }}
            >
              {/* Logo with 3D Effect */}
              <motion.div
                className="mb-12 relative"
                initial={{ opacity: 0, y: -50, rotateX: -30 }}
                animate={{ 
                  opacity: 1, 
                  y: 0, 
                  rotateX: 0,
                  filter: ["drop-shadow(0 0 20px rgba(16, 185, 129, 0.7))", "drop-shadow(0 0 10px rgba(16, 185, 129, 0.3))", "drop-shadow(0 0 20px rgba(16, 185, 129, 0.7))"],
                }}
                transition={{ 
                  type: "spring",
                  stiffness: 100,
                  damping: 15,
                  delay: 0.2,
                  filter: {
                    duration: 4,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }
                }}
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-green-300 to-blue-200 rounded-full -z-10 blur-2xl"
                  animate={{ 
                    opacity: [0.5, 0.8, 0.5],
                    scale: [0.8, 1.1, 0.8],
                  }}
                  transition={{ 
                    duration: 4, 
                    repeat: Infinity, 
                    ease: "easeInOut" 
                  }}
                />
                <img 
                  src="/logoQN.png" 
                  alt="QuantAI, NZ Logo" 
                  className="h-48 object-contain relative z-10 drop-shadow-xl"
                />
              </motion.div>
              
              {/* Title with Animated Gradient */}
              <motion.div
                ref={gradientTextRef}
                className="text-6xl font-black mb-6 tracking-tight relative"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  type: "spring",
                  stiffness: 100,
                  damping: 15,
                  delay: 0.4 
                }}
                style={{
                  background: 'linear-gradient(-45deg, #22d3ee, #10b981, #3b82f6, #8b5cf6)',
                  backgroundSize: '400% 400%',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                <motion.span
                  className="absolute -inset-1 -z-10 blur-md"
                  style={{
                    background: 'linear-gradient(-45deg, #22d3ee, #10b981, #3b82f6, #8b5cf6)',
                    backgroundSize: '400% 400%',
                    opacity: 0.4,
                  }}
                  animate={{ 
                    backgroundPosition: ['0% 0%', '100% 100%', '0% 0%'],
                  }}
                  transition={{ 
                    duration: 10, 
                    repeat: Infinity,
                    ease: "easeInOut" 
                  }}
                />
                QuantAI Hospital
              </motion.div>
              
              {/* Subtitle with Reveal Animation */}
              <motion.div
                className="text-2xl text-green-700 font-medium mb-8 relative"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                <motion.span
                  animate={{ 
                    textShadow: ["0 0 8px rgba(16, 185, 129, 0.7)", "0 0 2px rgba(16, 185, 129, 0.3)", "0 0 8px rgba(16, 185, 129, 0.7)"]
                  }}
                  transition={{ 
                    duration: 3, 
                    repeat: Infinity,
                    ease: "easeInOut" 
                  }}
                >
                  Voice Assistant by QuantAI, NZ
                </motion.span>
              </motion.div>
              
              {/* PoC Disclaimer with Glass Effect */}
              <motion.div
                className="bg-white/30 backdrop-blur-xl rounded-2xl px-10 py-6 max-w-xl text-center mb-8 shadow-xl border border-white/20"
                initial={{ opacity: 0, y: 30, rotateX: 20 }}
                animate={{ opacity: 1, y: 0, rotateX: 0 }}
                transition={{ 
                  type: "spring",
                  stiffness: 100,
                  damping: 15,
                  delay: 0.8 
                }}
                whileHover={{ scale: 1.02, boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)" }}
              >
                <motion.div 
                  className="text-amber-600 font-bold mb-3 text-xl"
                  animate={{ 
                    textShadow: ["0 0 0px rgba(217, 119, 6, 0)", "0 0 10px rgba(217, 119, 6, 0.5)", "0 0 0px rgba(217, 119, 6, 0)"]
                  }}
                  transition={{ 
                    duration: 2, 
                    repeat: Infinity,
                    ease: "easeInOut" 
                  }}
                >
                  Proof of Concept
                </motion.div>
                <p className="text-gray-800 text-base">
                  This is a QuantAI, NZ Proof of Concept application. It is not intended for production use, 
                  clinical decision making, or medical advice. All data and interactions are for demonstration purposes only.
                </p>
              </motion.div>
              
              {/* Server Status with Animated Indicator */}
              <motion.div
                className="flex items-center gap-4 bg-white/30 backdrop-blur-xl rounded-full px-6 py-3 shadow-lg border border-white/20"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
                whileHover={{ scale: 1.05 }}
              >
                <motion.span 
                  className={`w-3 h-3 rounded-full ${
                    serverStatus.status === 'online' ? 'bg-green-500' : 
                    serverStatus.status === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                  }`}
                  animate={{ 
                    boxShadow: [
                      `0 0 0px ${serverStatus.status === 'online' ? '#10b981' : serverStatus.status === 'offline' ? '#ef4444' : '#f59e0b'}`,
                      `0 0 10px ${serverStatus.status === 'online' ? '#10b981' : serverStatus.status === 'offline' ? '#ef4444' : '#f59e0b'}`,
                      `0 0 0px ${serverStatus.status === 'online' ? '#10b981' : serverStatus.status === 'offline' ? '#ef4444' : '#f59e0b'}`
                    ]
                  }}
                  transition={{ 
                    duration: 2, 
                    repeat: Infinity,
                    ease: "easeInOut" 
                  }}
                />
                <span className="text-base font-medium text-gray-700">Server Status: {serverStatus.status}</span>
                <span className="text-sm bg-white/50 px-3 py-1 rounded-full text-gray-600">v{serverStatus.version}</span>
              </motion.div>
              
              {/* Loading Indicator */}
              <motion.div
                className="mt-10 flex items-center gap-4"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.2 }}
              >
                <motion.div 
                  className="relative w-10 h-10"
                  animate={{ rotate: 360 }}
                  transition={{ 
                    duration: 3,
                    repeat: Infinity,
                    ease: "linear" 
                  }}
                >
                  <motion.span 
                    className="absolute inset-0 rounded-full border-3 border-transparent border-t-green-500"
                    animate={{ opacity: [1, 0.2, 1] }}
                    transition={{ 
                      duration: 1.5,
                      repeat: Infinity,
                      ease: "easeInOut" 
                    }}
                  />
                  <motion.span 
                    className="absolute inset-1 rounded-full border-3 border-transparent border-t-teal-400"
                    animate={{ 
                      rotate: -360,
                      opacity: [0.8, 0.1, 0.8]
                    }}
                    transition={{ 
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut" 
                    }}
                  />
                </motion.div>
                <motion.span 
                  className="text-base text-gray-600"
                  animate={{ opacity: [1, 0.5, 1] }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut" 
                  }}
                >
                  Loading your experience...
                </motion.span>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Parallax/animated backgrounds for 3D effect */}
      <div ref={parallaxRefs[5]} className="absolute inset-0 pointer-events-none z-0" style={{ background: 'radial-gradient(circle at 90% 90%, #a7f3d0 0%, transparent 70%)', opacity: 0.2 }} />
      <div ref={parallaxRefs[4]} className="absolute inset-0 pointer-events-none z-0" style={{ background: 'radial-gradient(circle at 50% 10%, #bbf7d0 0%, transparent 70%)', opacity: 0.2 }} />
      <div ref={parallaxRefs[3]} className="absolute inset-0 pointer-events-none z-0" style={{ background: 'radial-gradient(circle at 10% 10%, #a7f3d0 0%, transparent 70%)', opacity: 0.3 }} />
      <div ref={parallaxRefs[2]} className="absolute inset-0 pointer-events-none z-0" style={{ background: 'radial-gradient(circle at 80% 20%, #bbf7d0 0%, transparent 70%)', opacity: 0.5 }} />
      <div ref={parallaxRefs[1]} className="absolute inset-0 pointer-events-none z-0" style={{ background: 'radial-gradient(circle at 20% 80%, #6ee7b7 0%, transparent 70%)', opacity: 0.4 }} />
      <div ref={parallaxRefs[0]} className="absolute inset-0 pointer-events-none z-0" style={{ background: 'radial-gradient(circle at 60% 30%, #bbf7d0 0%, transparent 70%)', opacity: 0.7 }} />
      {/* Floating animated shapes */}
      {floatingShapes.map((shape, i) => (
        <motion.div
          key={i}
          className={`absolute z-0 rounded-full ${shape.className}`}
          style={{ width: shape.size, height: shape.size, top: shape.top, left: shape.left, opacity: 0.5 }}
          animate={{ y: [0, -10, 0] }}
          transition={{ repeat: Infinity, duration: 3 + i, delay: shape.delay, ease: 'easeInOut' }}
        />
      ))}

      {/* Enhanced Header with Glassmorphism */}
      <motion.div
        className="flex flex-col items-center mt-12 mb-8 relative z-10 glass-card p-8 w-full max-w-2xl"
        initial={{ opacity: 0, y: -40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
      >
        {/* QuantAI Logo */}
        <motion.div
          className="mb-4 flex flex-col items-center"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
        >
          <img 
            src="/logoQN.png" 
            alt="QuantAI, NZ Logo" 
            className="h-16 object-contain mb-2" 
          />
          <div className="flex items-center gap-2">
            <div className="text-xs px-2 py-0.5 bg-amber-100 text-amber-800 rounded-full">Proof of Concept</div>
            <div className="text-xs text-gray-500">Not for clinical use</div>
          </div>
        </motion.div>
        
        <h1 className="text-2xl md:text-3xl font-semibold text-center text-black mb-2 moving-gradient text-gradient">Kia ora! Welcome to QuantAI Hospital</h1>
        <h2 className="text-xl md:text-2xl font-medium text-center text-gray-700">How can I assist you today?</h2>
        <p className="text-gray-400 text-sm mt-2">Choose a prompt below or write your own to start chatting with QuantAI Hospital</p>
      </motion.div>

      {/* Language Selector with Fixed Positioning */}
      <motion.div 
        className="relative z-50 mb-6 flex justify-center w-full max-w-xl"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="relative">
          <button
            className={`glass-button flex items-center px-6 py-3 text-green-700 text-sm font-medium rounded-xl backdrop-blur-md bg-white/30 hover:bg-white/40 transition-all duration-200 ${
              showLangDropdown ? 'shadow-glow' : ''
            }`}
            onClick={() => setShowLangDropdown((v) => !v)}
            aria-haspopup="listbox"
            aria-expanded={showLangDropdown}
            aria-label="Select language"
          >
            {languageIcons[selectedLanguage] || <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#a7f3d0" /></svg>}
            {selectedLanguage.charAt(0).toUpperCase() + selectedLanguage.slice(1)}
            <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
          </button>
          <AnimatePresence>
            {showLangDropdown && (
              <>
                <motion.div
                  className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={() => setShowLangDropdown(false)}
                />
                <motion.div
                  className="absolute right-0 mt-2 w-64 rounded-xl bg-white/90 backdrop-blur-md shadow-xl border border-gray-200 overflow-hidden z-50"
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.2, ease: "easeOut" }}
                >
                  <div className="max-h-72 overflow-y-auto py-2">
                    {languages.map((lang) => (
                      <motion.button
                        key={lang}
                        className={`w-full flex items-center px-4 py-3 cursor-pointer transition-colors duration-150 ${
                          selectedLanguage === lang 
                            ? 'bg-green-50 text-green-700' 
                            : 'hover:bg-gray-50 text-gray-700'
                        }`}
                        onClick={() => {
                          setSelectedLanguage(lang);
                          setShowLangDropdown(false);
                        }}
                        whileHover={{ x: 4 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      >
                        <div className="flex items-center w-full">
                          {languageIcons[lang] || <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#a7f3d0" /></svg>}
                          <span className="flex-1 text-left">{lang.charAt(0).toUpperCase() + lang.slice(1)}</span>
                          {selectedLanguage === lang && (
                            <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                      </motion.button>
                    ))}
                  </div>
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Prompt Buttons */}
      <motion.div
        className="flex flex-wrap gap-4 justify-center mb-8 relative z-10"
        initial="hidden"
        animate="visible"
        variants={{
          hidden: {},
          visible: {
            transition: { staggerChildren: 0.08 },
          },
        }}
      >
        {prompts.map((prompt, idx) => (
          <motion.button
            key={prompt}
            className="px-4 py-2 bg-white border border-gray-200 rounded-lg shadow hover:bg-green-50 transition text-gray-700 text-sm font-medium"
            onClick={() => setInput(prompt)}
            whileHover={{ scale: 1.07 }}
            whileTap={{ scale: 0.97 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + idx * 0.08, type: 'spring', stiffness: 200 }}
          >
            {prompt}
          </motion.button>
        ))}
      </motion.div>

      {/* Enhanced Chat Area with Glassmorphism */}
      <motion.div
        className="w-full max-w-2xl flex-1 flex flex-col glass-card p-8 mb-12 relative z-10 rounded-2xl border border-white/20 backdrop-blur-xl bg-white/30 shadow-xl"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut', delay: 0.3 }}
      >
        <div className="flex-1 overflow-y-auto mb-6 space-y-6 scrollbar-thin px-2">
          <AnimatePresence initial={false}>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                className={`flex flex-col ${msg.sender === 'bot' ? 'items-start' : 'items-end'} space-y-2`}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 20, scale: 0.95 }}
                transition={{ duration: 0.2 }}
              >
                <div 
                  className={`px-6 py-4 rounded-2xl max-w-[85%] shadow-sm ${
                    msg.sender === 'bot' 
                      ? 'bg-white/80 backdrop-blur-md text-gray-800 border border-gray-100' 
                      : 'bg-green-500 bg-opacity-90 backdrop-blur-sm text-white'
                  }`}
                >
                  {msg.text}
                </div>
                
                {/* Speaker button below the message for bot messages */}
                {msg.sender === 'bot' && (
                  <motion.button 
                    onClick={() => speakWithElevenLabs(msg.text, i)}
                    className={`mt-1 ml-1 flex items-center gap-2 px-3 py-1.5 ${
                      isSpeaking === i.toString() 
                        ? 'bg-green-100' 
                        : 'bg-white/80 backdrop-blur-sm'
                    } rounded-lg shadow-sm hover:bg-green-50 transition text-xs text-green-700 border border-gray-100`}
                    title="Listen with ElevenLabs"
                    whileHover={{ scale: isSpeaking === i.toString() ? 1 : 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    disabled={isSpeaking !== null}
                  >
                    {isSpeaking === i.toString() ? (
                      <>
                        <svg className="w-4 h-4 text-green-600 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                        </svg>
                        Speaking...
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 text-green-600">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 0 1 0 12.728M16.463 8.288a5.25 5.25 0 0 1 0 7.424M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
                        </svg>
                        Listen
                      </>
                    )}
                  </motion.button>
                )}
              </motion.div>
            ))}
            {loading && (
              <motion.div
                className="flex justify-start"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="bg-white/80 backdrop-blur-md px-6 py-4 rounded-2xl text-green-700 max-w-[85%] border border-gray-100 shadow-sm">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse delay-75" />
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse delay-150" />
                    <span className="text-sm">QuantAI Hospital is thinking...</span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Error Message */}
        {error && (
          <motion.div 
            className="text-red-500 text-sm mb-4 bg-red-50/80 backdrop-blur-sm p-4 rounded-xl border border-red-100"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {error}
          </motion.div>
        )}

        {/* Enhanced Input Area */}
        <form className="flex items-center gap-3" onSubmit={handleSubmit}>
          <div className="relative flex-1">
            <input
              className={`w-full px-6 py-4 pr-12 rounded-xl bg-white/80 backdrop-blur-md border border-gray-200 focus:border-green-500 focus:ring-2 focus:ring-green-200 transition-all duration-200 ${
                voiceMode ? 'text-green-900' : ''
              }`}
              placeholder="How can QuantAI Hospital help you today?"
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={listening || loading}
            />
          </div>
          
          <div className="flex items-center gap-3">
            <motion.button
              type="button"
              className={`p-3 rounded-xl bg-white/80 backdrop-blur-md border border-gray-200 hover:bg-white/90 transition-all duration-200 ${
                voiceMode ? 'shadow-glow border-green-200' : ''
              }`}
              onClick={() => setVoiceMode(v => !v)}
              disabled={loading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="relative flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className={`w-6 h-6 text-green-500 ${listening ? 'animate-pulse' : ''}`}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75v1.5m0 0h3m-3 0H9m6-6a3 3 0 11-6 0V6a3 3 0 016 0v7.5z" />
                </svg>
                {listening && (
                  <motion.span
                    className="absolute w-8 h-8 rounded-full border-2 border-green-300"
                    initial={{ scale: 0.8, opacity: 0.8 }}
                    animate={{ scale: 1.5, opacity: 0 }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                )}
              </span>
            </motion.button>

            {voiceMode ? (
              <motion.button
                type="button"
                className={`px-6 py-4 rounded-xl bg-white/80 backdrop-blur-md border border-gray-200 text-green-700 font-medium ${
                  listening || loading ? 'opacity-50' : 'hover:bg-white/90'
                } transition-all duration-200`}
                onClick={handleVoiceInput}
                disabled={listening || loading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {listening ? (
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                    </svg>
                    Listening...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24">
                      <path d="M12 4v16m8-8H4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                    Speak
                  </span>
                )}
              </motion.button>
            ) : (
              <motion.button
                type="submit"
                className="px-6 py-4 rounded-xl bg-white/80 backdrop-blur-md border border-gray-200 text-green-700 font-medium hover:bg-white/90 transition-all duration-200"
                disabled={loading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Send
              </motion.button>
            )}
          </div>
        </form>

        {/* Footer Info */}
        <div className="flex items-center gap-3 mt-6 text-sm text-gray-500">
          <img src="/logoQN.png" alt="QuantAI Logo" className="h-5 object-contain" />
          <span>QuantAI Hospital Voice Assistant</span>
          <span className="px-2 py-0.5 rounded-lg bg-amber-50/80 backdrop-blur-sm text-amber-700 text-xs">PoC</span>
        </div>
        <div className="text-gray-500 text-sm mt-3 bg-white/50 backdrop-blur-sm px-4 py-3 rounded-xl border border-gray-100">
          <strong className="text-amber-600">Disclaimer:</strong> This QuantAI Hospital Assistant is a Proof of Concept (PoC) 
          by QuantAI, NZ. Not intended for clinical decision making, medical advice, or production use.
        </div>
      </motion.div>

      {/* Enhanced Server Status Indicator */}
      <motion.div 
        className="fixed bottom-4 right-4 glass-card px-4 py-2 flex items-center gap-2"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2 }}
      >
        <motion.span 
          className={`w-2 h-2 rounded-full ${
            serverStatus.status === 'online' ? 'bg-green-500' : 
            serverStatus.status === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
          }`}
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <span className="text-sm text-gray-600">
          {serverStatus.status === 'online' ? 'Connected' : 
           serverStatus.status === 'offline' ? 'Disconnected' : 'Connecting...'}
        </span>
      </motion.div>

      {/* Keyboard Shortcut Hint */}
      <div className="text-gray-400 text-xs mb-4 relative z-10 glass-effect px-3 py-1 rounded-full">
        Use <kbd className="bg-white bg-opacity-20 px-1 rounded">shift + return</kbd> for new line
      </div>
    </div>
  );
}
