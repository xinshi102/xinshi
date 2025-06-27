// 首页管理类
class HomeManager {
    constructor() {
        this.initializeEventListeners();
    }

    // 初始化事件监听器
    initializeEventListeners() {
        // 为所有CSS类名为feature-card的元素绑定点击事件
        document.querySelectorAll('.feature-card').forEach(card => {
            card.addEventListener('click', () => {
                const link = card.getAttribute('data-link');
                if (link) {//页面跳转
                    window.location.href = link;
                }
            });
        });

        // 添加页面加载动画
        this.animateOnLoad();
    }

    // 页面加载动画
    animateOnLoad() {
        const welcomeSection = document.querySelector('.welcome-section');
        if (welcomeSection) {
            welcomeSection.style.opacity = '0';//完全透明
            welcomeSection.style.transform = 'translateY(20px)';//初始隐藏在下方
            
            setTimeout(() => {//延迟100ms触发动画
                welcomeSection.style.transition = 'all 0.5s ease';
                welcomeSection.style.opacity = '1';//完全可见
                welcomeSection.style.transform = 'translateY(0)';
            }, 100);
        }

        // 为功能卡片添加延迟动画
        document.querySelectorAll('.feature-card').forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.5s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100 * (index + 1));
        });
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new HomeManager();
}); 