import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Language {
  id: string;
  name: string;
  icon: string;
}

interface LanguageSelectorProps {
  onLanguageSelect: (language: Language) => void;
  selectedLanguage?: Language;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  onLanguageSelect,
  selectedLanguage
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLanguages = async () => {
      try {
        const response = await fetch('/languages');
        if (!response.ok) {
          throw new Error('Failed to fetch languages');
        }
        const data = await response.json();
        setLanguages(data);
        if (!selectedLanguage && data.length > 0) {
          onLanguageSelect(data[0]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    fetchLanguages();
  }, []);

  const dropdownVariants = {
    hidden: {
      opacity: 0,
      y: -10,
      scale: 0.95,
    },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration: 0.2,
        ease: 'easeOut',
      },
    },
    exit: {
      opacity: 0,
      y: -10,
      scale: 0.95,
      transition: {
        duration: 0.15,
        ease: 'easeIn',
      },
    },
  };

  if (isLoading) {
    return (
      <div className="relative inline-block">
        <div className="h-10 w-40 animate-pulse bg-gray-200 rounded-lg"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="relative inline-block">
        <div className="h-10 px-4 flex items-center text-red-500 bg-red-50 rounded-lg">
          Error loading languages
        </div>
      </div>
    );
  }

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-4 py-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200 border border-gray-200"
      >
        {selectedLanguage && (
          <>
            <img
              src={selectedLanguage.icon}
              alt={selectedLanguage.name}
              className="w-5 h-5"
            />
            <span className="text-gray-700">{selectedLanguage.name}</span>
          </>
        )}
        <svg
          className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${
            isOpen ? 'transform rotate-180' : ''
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="absolute mt-2 w-48 rounded-lg bg-white shadow-xl border border-gray-200 overflow-hidden z-50"
            >
              <div className="max-h-60 overflow-y-auto">
                {languages.map((language) => (
                  <button
                    key={language.id}
                    onClick={() => {
                      onLanguageSelect(language);
                      setIsOpen(false);
                    }}
                    className={`w-full flex items-center space-x-3 px-4 py-2 text-left hover:bg-gray-50 transition-colors duration-150 ${
                      selectedLanguage?.id === language.id
                        ? 'bg-purple-50 text-purple-700'
                        : 'text-gray-700'
                    }`}
                  >
                    <img
                      src={language.icon}
                      alt={language.name}
                      className="w-5 h-5"
                    />
                    <span>{language.name}</span>
                  </button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LanguageSelector; 