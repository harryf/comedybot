const isDevelopment = process.env.NODE_ENV !== 'production';

class Logger {
  static debug(...args) {
    if (isDevelopment) {
      console.debug(...args);
    }
  }

  static log(...args) {
    if (isDevelopment) {
      console.log(...args);
    }
  }

  static info(...args) {
    if (isDevelopment) {
      console.info(...args);
    }
  }

  static warn(...args) {
    
    console.warn(...args);
  }

  static error(...args) {
    // Always log errors, even in production
    console.error(...args);
  }
}

export default Logger;
