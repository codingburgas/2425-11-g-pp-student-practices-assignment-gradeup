// Main JavaScript file for School Recommendation System

document.addEventListener('DOMContentLoaded', function() {
    // Enable Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Enable Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Initialize Bootstrap Accordion for FAQ - Let Bootstrap handle it naturally
    // No custom initialization needed, Bootstrap will auto-initialize

    // Initialize all systems
    initMovingParticlesBackground();
    initHeroParticles();
    initWebParticles();
    initScrollAnimations();
    initThemeToggle();
    initSmoothScrolling();
    initMobileMenu();
    initCounterAnimations();
    initCursorTrail();
    initStickyNavbar();
    initTestimonialCarousel();
    // Remove custom FAQ init since Bootstrap handles it
    initFeatureTabs();
    initNotifications();
    
    // Check for flash messages
    checkFlashMessages();
    
    // Add initial animations to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

// ===== MOVING PARTICLES BACKGROUND SYSTEM =====
function initMovingParticlesBackground() {
    const container = document.getElementById('particlesBackground');
    if (!container) return;
    
    // Check if user prefers reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }
    
    const particleCount = window.innerWidth > 768 ? 40 : 20;
    
    for (let i = 0; i < particleCount; i++) {
        createBackgroundParticle(container);
    }
    
    // Continuously create new particles
    setInterval(() => {
        if (container.children.length < particleCount) {
            createBackgroundParticle(container);
        }
    }, 3000);
}

function createBackgroundParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // Random horizontal position
    particle.style.left = Math.random() * 100 + '%';
    
    // Random size between 2px and 6px
    const size = Math.random() * 4 + 2;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    // Random animation delay and duration
    particle.style.animationDelay = Math.random() * 5 + 's';
    particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
    
    // Random opacity
    particle.style.opacity = Math.random() * 0.6 + 0.2;
    
    container.appendChild(particle);
    
    // Remove particle after animation completes
    const animationDuration = parseFloat(particle.style.animationDuration) * 1000;
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
    }, animationDuration);
}

// ===== HERO BOX PARTICLES SYSTEM =====
function initHeroParticles() {
    const container = document.getElementById('heroParticles');
    if (!container) return;
    
    // Check if user prefers reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }
    
    const particleCount = 15;
    
    for (let i = 0; i < particleCount; i++) {
        createHeroParticle(container);
    }
    
    // Continuously create new particles
    setInterval(() => {
        if (container.children.length < particleCount) {
            createHeroParticle(container);
        }
    }, 2000);
}

function createHeroParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'hero-particle';
    
    // Random horizontal position
    particle.style.left = Math.random() * 100 + '%';
    
    // Random size between 3px and 8px
    const size = Math.random() * 5 + 3;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    // Random animation delay and duration
    particle.style.animationDelay = Math.random() * 3 + 's';
    particle.style.animationDuration = (Math.random() * 6 + 12) + 's';
    
    // Random opacity
    particle.style.opacity = Math.random() * 0.8 + 0.2;
    
    container.appendChild(particle);
    
    // Remove particle after animation completes
    const animationDuration = parseFloat(particle.style.animationDuration) * 1000;
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
    }, animationDuration);
}

// ===== WEB PARTICLES SYSTEM FOR AUTH PAGES =====
function initWebParticles() {
    const container = document.getElementById('webParticles');
    if (!container) return;
    
    // Check if user prefers reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }
    
    const particleCount = window.innerWidth > 768 ? 60 : 30;
    
    // Create floating particles
    for (let i = 0; i < particleCount; i++) {
        createWebParticle(container);
    }
    
    // Create connecting lines
    for (let i = 0; i < 15; i++) {
        createWebLine(container);
    }
    
    // Continuously create new particles
    setInterval(() => {
        if (container.querySelectorAll('.web-particle').length < particleCount) {
            createWebParticle(container);
        }
        if (container.querySelectorAll('.web-line').length < 15) {
            createWebLine(container);
        }
    }, 2000);
}

function createWebParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'web-particle';
    
    // Random horizontal position
    particle.style.left = Math.random() * 100 + '%';
    
    // Random size between 2px and 5px
    const size = Math.random() * 3 + 2;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    // Random animation delay and duration
    particle.style.animationDelay = Math.random() * 5 + 's';
    particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
    
    // Random color variation
    const colors = [
        'rgba(139, 92, 246, 0.8)',
        'rgba(6, 182, 212, 0.6)',
        'rgba(236, 72, 153, 0.7)',
        'rgba(255, 255, 255, 0.5)'
    ];
    particle.style.background = colors[Math.floor(Math.random() * colors.length)];
    
    container.appendChild(particle);
    
    // Remove particle after animation completes
    const animationDuration = parseFloat(particle.style.animationDuration) * 1000;
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
    }, animationDuration);
}

function createWebLine(container) {
    const line = document.createElement('div');
    line.className = 'web-line';
    
    // Random position
    line.style.left = Math.random() * 80 + 10 + '%';
    line.style.top = Math.random() * 80 + 10 + '%';
    
    // Random width
    line.style.width = Math.random() * 100 + 50 + 'px';
    
    // Random rotation
    line.style.transform = `rotate(${Math.random() * 360}deg)`;
    
    // Random animation delay
    line.style.animationDelay = Math.random() * 3 + 's';
    
    container.appendChild(line);
    
    // Remove line after 10 seconds
    setTimeout(() => {
        if (line.parentNode) {
            line.parentNode.removeChild(line);
        }
    }, 10000);
}

// ===== ENHANCED SCROLL ANIMATIONS =====
function initScrollAnimations() {
    const animatedElements = document.querySelectorAll('.animate-fade-in, .animate-slide-up, .animate-slide-left, .animate-slide-right, .animate-scale');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                const delay = element.getAttribute('data-delay') || 0;
                
                setTimeout(() => {
                    element.classList.add('in-view');
                }, parseInt(delay));
                
                // Unobserve after animation to prevent repeated triggers
                observer.unobserve(element);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    animatedElements.forEach(element => {
        observer.observe(element);
    });
}

// Theme Toggle
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('light-theme');
            const icon = this.querySelector('i');
            
            if (document.body.classList.contains('light-theme')) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
                localStorage.setItem('theme', 'light');
                showToast('Light theme activated', 'info', 2000);
            } else {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
                localStorage.setItem('theme', 'dark');
                showToast('Dark theme activated', 'info', 2000);
            }
        });
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            const icon = themeToggle.querySelector('i');
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        }
    }
}

// Smooth Scrolling
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                
                const offsetTop = target.offsetTop - 100; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Counter Animations
function initCounterAnimations() {
    let countersAnimated = false;
    
    const statNumbers = document.querySelectorAll('.stat-number');
    
    function animateCounters() {
        if (countersAnimated) return;
        
        statNumbers.forEach(stat => {
            const target = stat.textContent.replace(/\D/g, ''); // Extract numbers only
            if (target) {
                animateCounter(stat, 0, parseInt(target), 2000);
            }
        });
        
        countersAnimated = true;
    }
    
    function animateCounter(element, start, end, duration) {
        const range = end - start;
        const increment = end > start ? 1 : -1;
        const stepTime = Math.abs(Math.floor(duration / range));
        
        let current = start;
        const timer = setInterval(() => {
            current += increment;
            element.textContent = current + (element.textContent.includes('+') ? '+' : '');
            
            if (current === end) {
                clearInterval(timer);
            }
        }, stepTime);
    }
    
    // Trigger counter animation when stats section is visible
    const statsSection = document.querySelector('.stats-section');
    if (statsSection) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    setTimeout(animateCounters, 500);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });
        
        observer.observe(statsSection);
    }
}

// Mobile Menu
function initMobileMenu() {
    const mobileToggle = document.querySelector('.mobile-menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    if (mobileToggle && sidebar) {
        mobileToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
            document.body.classList.toggle('sidebar-open');
        });
        
        // Close sidebar when clicking outside
        document.addEventListener('click', function(e) {
            if (!sidebar.contains(e.target) && !mobileToggle.contains(e.target)) {
                sidebar.classList.remove('active');
                document.body.classList.remove('sidebar-open');
            }
        });
    }
}

// Cursor Trail Effect
function initCursorTrail() {
    if (window.innerWidth <= 768) return; // Disable on mobile
    
    let mouseTrail = [];
    const trailLength = 10;
    
    document.addEventListener('mousemove', function(e) {
        mouseTrail.push({ x: e.clientX, y: e.clientY });
        
        if (mouseTrail.length > trailLength) {
            mouseTrail.shift();
        }
        
        updateTrail();
    });
    
    function updateTrail() {
        // Remove existing trail elements
        document.querySelectorAll('.cursor-trail').forEach(el => el.remove());
        
        mouseTrail.forEach((point, index) => {
            if (index % 2 === 0) { // Only show every other point for performance
                createTrailPoint(point.x, point.y, index);
            }
        });
    }
    
    function createTrailPoint(x, y, index) {
        const trail = document.createElement('div');
        trail.className = 'cursor-trail';
        trail.style.cssText = `
            position: fixed;
            left: ${x}px;
            top: ${y}px;
            width: 4px;
            height: 4px;
            background: rgba(139, 92, 246, ${0.8 - (index * 0.08)});
            border-radius: 50%;
            pointer-events: none;
            z-index: 9999;
            transition: opacity 0.3s ease;
        `;
        
        document.body.appendChild(trail);
        
        setTimeout(() => {
            trail.style.opacity = '0';
            setTimeout(() => trail.remove(), 300);
        }, 100);
    }
}

// Sticky Navbar
function initStickyNavbar() {
    const navbar = document.querySelector('.top-navbar');
    if (!navbar) return;
    
    let lastScrollY = window.scrollY;
    
    window.addEventListener('scroll', throttle(() => {
        const currentScrollY = window.scrollY;
        
        if (currentScrollY > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScrollY = currentScrollY;
    }, 16));
}

// Testimonial Carousel
function initTestimonialCarousel() {
    const carousel = document.querySelector('#testimonialsCarousel');
    if (carousel) {
        // Auto-advance carousel
        setInterval(() => {
            const nextButton = carousel.querySelector('.carousel-control-next');
            if (nextButton) {
                nextButton.click();
            }
        }, 8000);
    }
}

// ===== SIMPLIFIED FAQ ACCORDION - LET BOOTSTRAP HANDLE IT =====
function initFAQAccordion() {
    // Bootstrap automatically handles accordion functionality
    // We just need to ensure the elements are properly structured
    // which they already are in the HTML
    
    // Optional: Add any custom animations or enhancements here
    const accordionButtons = document.querySelectorAll('.accordion-button');
    
    accordionButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Add a subtle animation class when clicked
            this.classList.add('accordion-clicking');
            setTimeout(() => {
                this.classList.remove('accordion-clicking');
            }, 150);
        });
    });
}

// Feature Tabs
function initFeatureTabs() {
    const tabButtons = document.querySelectorAll('.nav-tabs .nav-link');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Add smooth transition effect
            const targetPane = document.querySelector(this.getAttribute('data-bs-target'));
            if (targetPane) {
                targetPane.style.opacity = '0';
                setTimeout(() => {
                    targetPane.style.opacity = '1';
                }, 150);
            }
        });
    });
}

// UNIFIED NOTIFICATION SYSTEM FOR ENTIRE SITE
function showNotification(message, type = 'info', duration = 4000) {
    const notification = document.createElement('div');
    notification.className = `cosmic-notification cosmic-notification-${type}`;
    
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-triangle',
        warning: 'fas fa-exclamation-circle',
        info: 'fas fa-info-circle'
    };
    
    notification.innerHTML = `
        <div class="notification-content">
            <i class="${icons[type]} me-2"></i>
            <span class="notification-message">${message}</span>
        </div>
        <button type="button" class="notification-close" onclick="closeNotification(this.parentElement)">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to container
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    // Animate in
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Auto remove
    if (duration > 0) {
        setTimeout(() => closeNotification(notification), duration);
    }
    
    return notification;
}

function closeNotification(notification) {
    if (notification && notification.parentNode) {
        notification.classList.add('hiding');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}

// Toast Notifications (legacy support - now redirects to unified system)
function showToast(message, type = 'info', duration = 3000) {
    return showNotification(message, type, duration);
}

function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-triangle',
        warning: 'exclamation-circle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function closeToast(toast) {
    return closeNotification(toast);
}

// Notifications
function initNotifications() {
    // Check URL parameters for notifications
    const urlParams = new URLSearchParams(window.location.search);
    
    if (urlParams.get('logged_in') === 'true') {
        showToast('Successfully logged in!', 'success');
    }
    
    if (urlParams.get('logged_out') === 'true') {
        showToast('Successfully logged out!', 'info');
    }
    
    if (urlParams.get('registered') === 'true') {
        showToast('Account created successfully!', 'success');
    }
}

// Check Flash Messages - Updated to use unified system
function checkFlashMessages() {
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(alert => {
        const text = alert.textContent.trim();
        const type = alert.classList.contains('alert-danger') ? 'error' : 
                    alert.classList.contains('alert-success') ? 'success' :
                    alert.classList.contains('alert-warning') ? 'warning' : 'info';
        
        if (text) {
            showNotification(text, type);
        }
        
        // Hide the original alert
        alert.style.display = 'none';
    });
    
    // Test notification to ensure system works
    if (window.location.pathname === '/') {
        setTimeout(() => {
            showNotification('Welcome to GradeUP! Notification system is working.', 'info', 3000);
        }, 1000);
    }
}

// Utility Functions
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

function debounce(func, wait, immediate) {
    let timeout;
    return function() {
        const context = this, args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
} 