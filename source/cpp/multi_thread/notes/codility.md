Tesla
by
 
Tesla invites you to complete a technical assessment
access_time
80
minutes for
draft
1
task为帮助您顺利完成即将到来的技术笔试，以下是一些重要提示：
- 相关的cpp API
- std::mutex mMutex; mMutex.lock(); mMutex.unlock();
- std::timed_mutex mTimedMutex; mTimedMutex.try_lock_for(2ms);
- std::recursive_mutex mRecursiveMutex; mRecursiveMutex.lock(); mRecursiveMutex.unlock();

- std::lock_guard<std::mutex> lock(mMutex);
- std::scoped_lock<std::mutex> lock(mMutex1, mMutex2);
- std::unique_lock lock1{mMutex1, std::defer_lock}; std::unique_lock lock2{mMutex2, std::defer_lock}; std::lock(lock1, lock2);
- std::shared_lock<std::shared_mutex> lock1(mSharedMutex); std::unique_lock<std::shared_mutex> lock1(mSharedMutex);

- std::atomic_int aInt; aInt.fetch_add(x); aInt.fetch_sub(x); aInt.load(); aInt.store(x);

- std::condition_variable cv; cv.wait(mMutex, []{ return ready; }); cv.notify_one(); cv.notify_all();


技术准备：
•多线程编程：回顾多线程编程的核心思路，掌握线程安全的基本概念和同步机制。
•异常处理：熟悉C++中的异常处理机制，确保代码能够正确处理异常情况。
•边界条件：在实现代码后，充分测试各种边界条件和异常情况，确保代码的健壮性。
•C++17特性：熟悉C++17的高级特性，如线程库，原子操作，智能指针，并发工具等。

环境与流程：
•编译器版本：笔试支持C++17标准，如果使用本地编译器，请确保你的编译器版本兼容，避免因编译器问题导致的错误。
•限时完成：笔试限时80分钟完成，点击开始做题才会倒计时，倒计时结束会自动提交。请注意时间管理，建议先完成核心功能，预留时间进行编译和基础逻辑检查。
•环境准备：请提前确认好您的线上环境。您可以通过邮件中的链接进行练习，以确保您的设备和网络能够正常运行，如果网络缓慢可以通过VPN，换浏览器等方式，不要因为网速影响代码提交(练习的链接与笔试真正的题目没有相关性，只用于调试和适应笔试环境）

注意事项：
•独立完成：所有笔试环节需全程独立完成，禁止使用AI工具或其他外部辅助。
•白板规划：在笔试前，可以使用白板或草稿纸规划和设计代码，确保逻辑清晰。
•代码可读性：保证代码的可读性，使用规范的命名，为关键逻辑添加注释，这有助于梳理思路，也方便阅卷人理解。
•提交注意：每题不支持多次提交，请确保点击"run code"测试成功后再提交。提交前务必确认代码可以编译通过，避免因语法错误影响成绩。


希望这些建议能帮助你在笔试中发挥出色！祝你成功！
Before you begin
There's no option to pause. Make sure you will not be interrupted for 80 minutes.
Your solution(s) should consider all possible corner cases and handle large input efficiently. Passing the example test does not indicate that your solution is correct. The example test is not part of your final score.
After finishing the test you will receive feedback containing your score. See example feedback.
If you accidentally close your browser, use the invitation link to get back to your test.
You can write your solution(s) in C++.

FAQ arrow_right_alt
learn more about Codility tests

Feedback example arrow_right_alt
see an example of what you will receive after finishing the test

What behaviors will be tracked during the assessment?
keyboard
mouse
videocam
monitor
To protect the integrity of today’s assessment and ensure compliance with hiring policies, the hiring team has enabled features that track and record in-app behavior during the assessment.The actions you take in the Codility environment will be tracked, recorded, and made available to the hiring team. No decisions are made automatically from these recordings.
Do you need help?
If you require special arrangements to be made to take the test due to a disability, please inform your test sponsor and they will try to provide support.

In case of technical problems with your test, please contact support@codility.com.
Enable accessibility mode for screen readers
Are you ready?
I have read and accepted Codility's Terms of Service  and  Privacy Policy.

I will follow Codility’s Code of Honour and I understand that my solution will be automatically checked for plagiarism.
Take the demo testStart the test* indicates a required field
© 2009–2026 Codility Ltd., registered in England and Wales (No. 7048726). VAT ID GB981191408. Registered office: 107 Cheapside, London EC2V 6DN
This website uses cookies to enhance user experience and to analyze performance and traffic on our website. We also share information about your use of our site with our social media, advertising and analytics partners.
Cookie Settings Accept Cookies
