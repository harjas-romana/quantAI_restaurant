import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import gsap from 'gsap';

interface SplashScreenProps {
  onComplete: () => void;
}

const SplashScreen: React.FC<SplashScreenProps> = ({ onComplete }) => {
  useEffect(() => {
    // Animate and then call onComplete
    const timer = setTimeout(onComplete, 3000);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-gradient-to-br from-gray-50 to-white"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="relative">
        {/* Background Elements */}
        <motion.div
          className="absolute inset-0 -z-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        >
          <div className="absolute top-0 left-0 w-64 h-64 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob" />
          <div className="absolute top-0 right-0 w-64 h-64 bg-yellow-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000" />
          <div className="absolute -bottom-8 left-20 w-64 h-64 bg-pink-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000" />
        </motion.div>

        {/* Logo */}
        <motion.div
          className="mb-8 relative"
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{
            type: "spring",
            stiffness: 260,
            damping: 20,
            delay: 0.2
          }}
        >
          <div className="w-32 h-32 mx-auto relative">
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 animate-spin-slow" />
            <div className="absolute inset-1 rounded-full bg-white flex items-center justify-center">
              <span className="text-5xl font-bold bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 text-transparent bg-clip-text">
                Q
              </span>
            </div>
          </div>
        </motion.div>

        {/* Text Elements */}
        <motion.div
          className="text-center"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-transparent bg-clip-text">
            QuantAI Restaurant
          </h1>
          <p className="text-gray-600 mb-8">Your AI-powered culinary assistant</p>
        </motion.div>

        {/* Loading Indicator */}
        <motion.div
          className="flex justify-center items-center space-x-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
        >
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
        </motion.div>

        {/* Version Tag */}
        <motion.div
          className="absolute bottom-4 left-1/2 transform -translate-x-1/2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
        >
          <span className="px-3 py-1 text-xs text-purple-600 bg-purple-50 rounded-full">
            Version 1.0.0
          </span>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default SplashScreen; 