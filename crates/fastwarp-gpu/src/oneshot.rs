// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

struct State<T> {
    value: Option<T>,
    waker: Option<Waker>,
    sender_dropped: bool,
}

pub(crate) struct Sender<T> {
    state: Arc<Mutex<State<T>>>,
}

pub(crate) struct Receiver<T> {
    state: Arc<Mutex<State<T>>>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Canceled;

pub(crate) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let state = Arc::new(Mutex::new(State {
        value: None,
        waker: None,
        sender_dropped: false,
    }));
    (
        Sender {
            state: state.clone(),
        },
        Receiver { state },
    )
}

impl<T> Sender<T> {
    pub(crate) fn send(self, value: T) {
        let waker = {
            let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
            s.value = Some(value);
            s.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let waker = {
            let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
            s.sender_dropped = true;
            s.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

impl<T> Receiver<T> {
    pub(crate) fn try_recv(&self) -> Option<T> {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .value
            .take()
    }
}

impl<T> Future for Receiver<T> {
    type Output = Result<T, Canceled>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(v) = s.value.take() {
            return Poll::Ready(Ok(v));
        }
        if s.sender_dropped {
            return Poll::Ready(Err(Canceled));
        }
        s.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::task::Waker;

    fn poll_once<F: Future + Unpin>(f: &mut F) -> Poll<F::Output> {
        let mut cx = Context::from_waker(Waker::noop());
        Pin::new(f).poll(&mut cx)
    }

    #[test]
    fn a_sent_value_is_ready_on_the_next_poll() {
        let (tx, mut rx) = channel();
        assert!(poll_once(&mut rx).is_pending());
        tx.send(7u32);
        assert_eq!(poll_once(&mut rx), Poll::Ready(Ok(7)));
    }

    #[test]
    fn a_dropped_sender_cancels() {
        let (tx, mut rx) = channel::<u32>();
        drop(tx);
        assert_eq!(poll_once(&mut rx), Poll::Ready(Err(Canceled)));
    }

    #[test]
    fn try_recv_takes_the_value() {
        let (tx, rx) = channel();
        assert_eq!(rx.try_recv(), None);
        tx.send(1u8);
        assert_eq!(rx.try_recv(), Some(1));
        assert_eq!(rx.try_recv(), None);
    }
}
